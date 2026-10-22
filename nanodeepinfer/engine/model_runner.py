import torch

from nanodeepinfer.config import Config
from nanodeepinfer.engine.sequence import Sequence
from nanodeepinfer.layers.sampler import Sampler
from nanodeepinfer.models.deepseek import DeepseekForCausalLM
from nanodeepinfer.utils.context import Context, SeqMeta, reset_context, set_context
from nanodeepinfer.utils.loader import load_weights


class ModelRunner:
    """单机版 Worker：备输入 -> 前向 -> 采样，对应原版 worker/model_runner.py 的主干。

    decode 走整批静态形状前向并以 CUDA graph 捕获重放（CPU 或 enforce_eager 时
    回退同一前向的 eager 执行）；prefill 走 Context 路径，支持前缀缓存与分块。
    """

    def __init__(self, config: Config):
        self.config = config
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        if config.dtype == "auto":
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        else:
            dtype = getattr(torch, config.dtype)
        if self.device.type == "cuda":
            cap = torch.cuda.get_device_capability(self.device)
            print(f"[nano-deepinfer] device={torch.cuda.get_device_name(self.device)} SM{cap[0]}{cap[1]} dtype={dtype}")
        self.model = DeepseekForCausalLM(config.hf_config, config.max_model_len).to(dtype)
        load_weights(self.model, config.model)
        self.model.to(self.device).eval()
        hash_map = getattr(self.model.model, "hash_map", None)
        if hash_map is not None:
            hash_map.to(self.device)
        self.ngram_k = config.hf_config.engram_max_ngram_size if config.hf_config.engram_layer_ids else 0
        self.sampler = Sampler()
        self.block_size = config.kvcache_block_size
        self.kv_caches = self._allocate_kv_cache(config, dtype)
        self.graphs: dict[tuple[int, int], tuple] = {}

    def _allocate_kv_cache(self, config: Config, dtype: torch.dtype) -> list[torch.Tensor]:
        cfg = config.hf_config
        latent = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        num_blocks = config.num_kvcache_blocks
        if num_blocks <= 0:
            if self.device.type == "cuda":
                free, _ = torch.cuda.mem_get_info()
                budget = int(free * config.gpu_memory_utilization)
            else:
                budget = 512 * 1024 ** 2
            per_block = self.block_size * latent * dtype.itemsize * cfg.num_hidden_layers
            num_blocks = max(64, budget // per_block)
        config.num_kvcache_blocks = num_blocks
        self.scratch_slot = num_blocks * self.block_size  # 额外一块草稿区，承接整批 prefill 的填充行
        shape = (num_blocks + 1, self.block_size, latent)
        # 必须零初始化：被注意力 mask 掉的位置仍会参与矩阵乘，未写入槽位若是 NaN 会经 0*NaN 污染结果
        return [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(cfg.num_hidden_layers)]

    def _slots(self, seq: Sequence, start: int, length: int) -> list[int]:
        slots = []
        for pos in range(start, start + length):
            block = seq.block_table[pos // self.block_size]
            slots.append(block * self.block_size + pos % self.block_size)
        return slots

    def _ngram_window(self, token_ids: list[int], start: int, length: int) -> list[list[int]]:
        """每调度位置的 n-gram 窗口（以当前 token 结尾、左补 0），对应原版 engram 哈希输入。"""
        k = self.ngram_k
        return [([0] * (k - p - 1) + token_ids[:p + 1])[-k:] for p in range(start, start + length)]

    def _ngram_tensor(self, windows: list) -> torch.Tensor | None:
        if not self.ngram_k:
            return None
        return torch.tensor(windows, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int | None]:
        if is_prefill:
            return self._run_prefill(seqs)
        if self.device.type == "cuda" and not self.config.enforce_eager:
            logits = self._run_decode_graph(seqs)
        else:
            logits = self._run_decode_eager(seqs)
        return self.sampler.forward(logits, [s.sampling_params for s in seqs])

    @torch.inference_mode()
    def _run_prefill(self, seqs: list[Sequence]) -> list[int | None]:
        if all(s.num_cached_tokens + s.num_scheduled_tokens == s.num_tokens for s in seqs):
            return self._run_prefill_batched(seqs)
        input_ids, positions, slots, metas, last_mask, ngrams = [], [], [], [], [], []
        sample_seqs: list[Sequence] = []
        for seq in seqs:
            start, length = seq.num_cached_tokens, seq.num_scheduled_tokens
            input_ids.extend(seq.token_ids[start: start + length])
            positions.extend(range(start, start + length))
            slots.extend(self._slots(seq, start, length))
            metas.append(SeqMeta(seq.block_table, start, length))
            if self.ngram_k:
                ngrams.extend(self._ngram_window(seq.token_ids, start, length))
            done = start + length == seq.num_tokens
            last_mask.extend([False] * (length - 1) + [done])
            if done:
                sample_seqs.append(seq)
        t = lambda v: torch.tensor(v, device=self.device)
        set_context(Context(True, metas, t(slots), self._ngram_tensor(ngrams)))
        logits = self.model(t(input_ids), t(positions), self.kv_caches, t(last_mask))
        reset_context()
        sampled = self.sampler.forward(logits, [s.sampling_params for s in sample_seqs])
        result: list[int | None] = [None] * len(seqs)
        it = iter(sampled)
        for i, seq in enumerate(seqs):
            if seq in sample_seqs:
                result[i] = next(it)
        return result

    @torch.inference_mode()
    def _run_prefill_batched(self, seqs: list[Sequence]) -> list[int | None]:
        bs, smax = len(seqs), max(s.num_scheduled_tokens for s in seqs)
        max_blocks = max(len(s.block_table) for s in seqs)
        input_ids = [[0] * smax for _ in range(bs)]
        positions = [[0] * smax for _ in range(bs)]
        slots = [[self.scratch_slot] * smax for _ in range(bs)]
        block_tables = [[0] * max_blocks for _ in range(bs)]
        ngrams = [[[0] * self.ngram_k for _ in range(smax)] for _ in range(bs)] if self.ngram_k else None
        for i, s in enumerate(seqs):
            start, length = s.num_cached_tokens, s.num_scheduled_tokens
            input_ids[i][:length] = s.token_ids[start: start + length]
            positions[i][:length] = range(start, start + length)
            if length < smax:
                positions[i][length:] = [start + length - 1] * (smax - length)
            slots[i][:length] = self._slots(s, start, length)
            block_tables[i][: len(s.block_table)] = s.block_table
            if self.ngram_k:
                ngrams[i][:length] = self._ngram_window(s.token_ids, start, length)
        t = lambda v: torch.tensor(v, device=self.device)
        logits = self.model.forward_prefill_batched(
            t(input_ids), t(positions), t(block_tables),
            t([s.num_cached_tokens for s in seqs]), t([s.num_scheduled_tokens for s in seqs]),
            t(slots), self.kv_caches, ngram_ids=self._ngram_tensor(ngrams) if ngrams else None)
        return self.sampler.forward(logits, [s.sampling_params for s in seqs])

    def _decode_inputs(self, seqs: list[Sequence]):
        bs = len(seqs)
        max_blocks = max(len(s.block_table) for s in seqs)
        bucket_blocks = 1 << (max_blocks - 1).bit_length()
        input_ids = [s.last_token for s in seqs]
        positions = [s.num_cached_tokens for s in seqs]
        context_lens = [p + 1 for p in positions]
        slots = [s.block_table[p // self.block_size] * self.block_size + p % self.block_size
                 for s, p in zip(seqs, positions)]
        block_tables = [[0] * bucket_blocks for _ in range(bs)]
        for i, s in enumerate(seqs):
            block_tables[i][: len(s.block_table)] = s.block_table
        ngrams = None
        if self.ngram_k:
            k = self.ngram_k
            ngrams = [[0] * (k - len(s.token_ids)) + s.token_ids if len(s.token_ids) < k
                      else s.token_ids[-k:] for s in seqs]
        return bucket_blocks, input_ids, positions, context_lens, slots, block_tables, ngrams

    @torch.inference_mode()
    def _run_decode_eager(self, seqs: list[Sequence]) -> torch.Tensor:
        _, input_ids, positions, context_lens, slots, block_tables, ngrams = self._decode_inputs(seqs)
        t = lambda v: torch.tensor(v, device=self.device)
        return self.model.forward_decode(t(input_ids), t(positions), t(block_tables), t(context_lens),
                                         t(slots), self.kv_caches, ngram_ids=self._ngram_tensor(ngrams))

    @torch.inference_mode()
    def _run_decode_graph(self, seqs: list[Sequence]) -> torch.Tensor:
        bucket_blocks, input_ids, positions, context_lens, slots, block_tables, ngrams = self._decode_inputs(seqs)
        key = (len(seqs), bucket_blocks)
        if key not in self.graphs:
            self._capture(key, bucket_blocks)
        graph, buffers, logits = self.graphs[key]
        t = lambda v: torch.tensor(v, device=self.device)
        buffers["input_ids"].copy_(t(input_ids))
        buffers["positions"].copy_(t(positions))
        buffers["context_lens"].copy_(t(context_lens))
        buffers["slot_mapping"].copy_(t(slots))
        buffers["block_tables"].copy_(t(block_tables))
        if ngrams is not None:
            buffers["ngram_ids"].copy_(t(ngrams))
        graph.replay()
        return logits

    def _capture(self, key: tuple[int, int], bucket_blocks: int):
        bs = key[0]
        buffers = {
            "input_ids": torch.zeros(bs, dtype=torch.long, device=self.device),
            "positions": torch.zeros(bs, dtype=torch.long, device=self.device),
            "context_lens": torch.ones(bs, dtype=torch.long, device=self.device),
            # warmup/capture 必须写草稿区而非槽 0，否则污染已分配块的 KV cache
            "slot_mapping": torch.full((bs,), self.scratch_slot, dtype=torch.long, device=self.device),
            "block_tables": torch.zeros(bs, bucket_blocks, dtype=torch.long, device=self.device),
        }
        args = (buffers["input_ids"], buffers["positions"], buffers["block_tables"],
                buffers["context_lens"], buffers["slot_mapping"], self.kv_caches)
        if self.ngram_k:
            buffers["ngram_ids"] = torch.zeros(bs, self.ngram_k, dtype=torch.long, device=self.device)
            args = args + (buffers["ngram_ids"],)
        fn = self.model.forward_decode
        for _ in range(3):
            fn(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = fn(*args)
        self.graphs[key] = (graph, buffers, logits)
