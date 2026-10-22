import torch

from nanodeepinfer.config import Config
from nanodeepinfer.engine.sequence import Sequence
from nanodeepinfer.layers.sampler import Sampler
from nanodeepinfer.models.deepseek import DeepseekForCausalLM
from nanodeepinfer.utils.context import Context, SeqMeta, reset_context, set_context
from nanodeepinfer.utils.loader import load_weights


class ModelRunner:
    """单机版 Worker：备输入 -> 前向 -> 采样，对应原版 worker/model_runner.py 的主干。

    原版在此之上叠了 CUDA graph、DP/EP/CP 通信与双后端 kernel 分发；
    教学版只保留一台设备上的等价数据流，MLA 用纯 PyTorch 复刻 FlashMLA 的数学。
    """

    def __init__(self, config: Config):
        self.config = config
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda" and config.profile == "rtx3060":
            cap = torch.cuda.get_device_capability(self.device)
            assert cap >= (8, 6), f"rtx3060 profile requires SM86+, got SM{cap[0]}{cap[1]}"
        if config.dtype == "auto":
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        else:
            dtype = getattr(torch, config.dtype)
        if self.device.type == "cuda":
            cap = torch.cuda.get_device_capability(self.device)
            name = torch.cuda.get_device_name(self.device)
            print(f"[nano-deepinfer] profile={config.profile} device={name} SM{cap[0]}{cap[1]} dtype={dtype}")
        self.model = DeepseekForCausalLM(config.hf_config, config.max_model_len).to(dtype)
        load_weights(self.model, config.model)
        self.model.to(self.device).eval()
        self.sampler = Sampler()
        self.block_size = config.kvcache_block_size
        self.kv_caches = self._allocate_kv_cache(config, dtype)

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
        shape = (num_blocks, self.block_size, latent)
        return [torch.empty(shape, dtype=dtype, device=self.device) for _ in range(cfg.num_hidden_layers)]

    def _slots(self, seq: Sequence, start: int, length: int) -> list[int]:
        slots = []
        for pos in range(start, start + length):
            block = seq.block_table[pos // self.block_size]
            slots.append(block * self.block_size + pos % self.block_size)
        return slots

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int | None]:
        input_ids, positions, slots, metas, last_mask = [], [], [], [], []
        sample_seqs: list[Sequence] = []
        for seq in seqs:
            start = seq.num_cached_tokens
            length = seq.num_scheduled_tokens
            if is_prefill:
                input_ids.extend(seq.token_ids[start: start + length])
            else:
                input_ids.append(seq.last_token)
            positions.extend(range(start, start + length))
            slots.extend(self._slots(seq, start, length))
            metas.append(SeqMeta(seq.block_table, start, length))
            done = (start + length == seq.num_tokens) if is_prefill else True
            last_mask.extend([False] * (length - 1) + [done])
            if done:
                sample_seqs.append(seq)
        t = lambda v: torch.tensor(v, device=self.device)
        set_context(Context(is_prefill, metas, t(slots)))
        logits = self.model(t(input_ids), t(positions), self.kv_caches, t(last_mask))
        reset_context()
        sampled = self.sampler.forward(logits, [s.sampling_params for s in sample_seqs])
        result: list[int | None] = [None] * len(seqs)
        it = iter(sampled)
        for i, seq in enumerate(seqs):
            if seq in sample_seqs:
                result[i] = next(it)
        return result
