import torch
from torch import nn

from nanodeepinfer.config import ModelConfig
from nanodeepinfer.layers.attention import MLAAttention
from nanodeepinfer.layers.attention_v4 import V4SparseAttention
from nanodeepinfer.layers.engram import Engram, NgramHash
from nanodeepinfer.layers.hc import HCHead, HCMix
from nanodeepinfer.layers.layernorm import RMSNorm
from nanodeepinfer.layers.moe import MLP, MoE
from nanodeepinfer.utils.context import Context, get_context, set_context


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int, max_len: int):
        super().__init__()
        self.self_attn = MLAAttention(cfg, max_len)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        is_moe = layer_id >= cfg.first_k_dense_replace and layer_id % cfg.moe_layer_freq == 0 and cfg.n_routed_experts > 0
        self.mlp = MoE(cfg) if is_moe else MLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions, kv_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def decode(self, x: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
               context_lens: torch.Tensor, slot_mapping: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.decode(self.input_layernorm(x), positions, block_tables, context_lens, slot_mapping, kv_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def prefill_batched(self, x: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
                        cached_lens: torch.Tensor, new_lens: torch.Tensor, slot_mapping: torch.Tensor,
                        kv_cache: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.prefill_batched(self.input_layernorm(x), positions, block_tables, cached_lens, new_lens, slot_mapping, kv_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class V4DecoderLayer(nn.Module):
    """V4 层：mHC 多路残差 + （稀疏）注意力 + MoE，Engram 层在 attn 前向 trunk 注入记忆。

    对应原版 decoder_layer.DeepseekDecoderLayer：每个子层 pre 把 trunk 混成单路、
    norm 后进子层，post 把输出按混合系数写回 trunk（原版 pre/post 融合 kernel
    这里拆成顺序调用，数学等价）。
    """

    def __init__(self, cfg: ModelConfig, layer_id: int, max_len: int, hash_map: NgramHash | None):
        super().__init__()
        self.self_attn = V4SparseAttention(cfg, max_len) if cfg.index_topk > 0 else MLAAttention(cfg, max_len)
        is_moe = layer_id >= cfg.first_k_dense_replace and layer_id % cfg.moe_layer_freq == 0 and cfg.n_routed_experts > 0
        self.mlp = MoE(cfg) if is_moe else MLP(cfg.hidden_size, cfg.intermediate_size)
        self.attn_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.hc_attn = HCMix(cfg, layer_id)
        self.hc_ffn = HCMix(cfg, layer_id)
        self.engram = None
        if cfg.engram_layer_ids and layer_id in cfg.engram_layer_ids:
            self.engram = Engram(cfg, cfg.engram_layer_ids.index(layer_id), hash_map)

    def _sublayer(self, trunk: torch.Tensor, hc: HCMix, norm: RMSNorm,
                  fn, ngram_ids: torch.Tensor | None) -> torch.Tensor:
        if self.engram is not None and hc is self.hc_attn and ngram_ids is not None:
            trunk = self.engram(trunk, ngram_ids)
        h, cached = hc.pre(trunk)
        y = fn(norm(h))
        return HCMix.post(y, trunk, cached)

    def forward(self, trunk: torch.Tensor, positions: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        ngram_ids = get_context().ngram_ids
        trunk = self._sublayer(trunk, self.hc_attn, self.attn_layernorm,
                               lambda h: self.self_attn(h, positions, kv_cache), ngram_ids)
        return self._sublayer(trunk, self.hc_ffn, self.ffn_layernorm, self.mlp, ngram_ids)

    def decode(self, trunk: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
               context_lens: torch.Tensor, slot_mapping: torch.Tensor, ngram_ids: torch.Tensor | None,
               kv_cache: torch.Tensor) -> torch.Tensor:
        trunk = self._sublayer(trunk, self.hc_attn, self.attn_layernorm,
                               lambda h: self.self_attn.decode(h, positions, block_tables, context_lens, slot_mapping, kv_cache), ngram_ids)
        return self._sublayer(trunk, self.hc_ffn, self.ffn_layernorm, self.mlp, ngram_ids)

    def prefill_batched(self, trunk: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
                        cached_lens: torch.Tensor, new_lens: torch.Tensor, slot_mapping: torch.Tensor,
                        ngram_ids: torch.Tensor | None, kv_cache: torch.Tensor) -> torch.Tensor:
        trunk = self._sublayer(trunk, self.hc_attn, self.attn_layernorm,
                               lambda h: self.self_attn.prefill_batched(h, positions, block_tables, cached_lens, new_lens, slot_mapping, kv_cache), ngram_ids)
        return self._sublayer(trunk, self.hc_ffn, self.ffn_layernorm, self.mlp, ngram_ids)


class DeepseekModel(nn.Module):
    def __init__(self, cfg: ModelConfig, max_len: int):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        if cfg.is_v4:
            assert cfg.hc_enabled, "V4 层要求启用 mHC（与原版 MoeDecoderBlockHC 一致）"
            self.hash_map = NgramHash(cfg, torch.device("cpu")) if cfg.engram_layer_ids else None
            self.layers = nn.ModuleList([V4DecoderLayer(cfg, i, max_len, self.hash_map) for i in range(cfg.num_hidden_layers)])
            self.hc_head = HCHead(cfg)
        else:
            self.layers = nn.ModuleList([DecoderLayer(cfg, i, max_len) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(x).unsqueeze(-2).expand(*x.shape, self.cfg.hc_mult, self.cfg.hidden_size).contiguous()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: list[torch.Tensor],
                ngram_ids: torch.Tensor | None = None) -> torch.Tensor:
        if ngram_ids is not None:  # Context 路径：engram 从全局上下文取 n-gram 窗口
            ctx = get_context()
            set_context(Context(ctx.is_prefill, ctx.seq_metas, ctx.slot_mapping, ngram_ids))
        x = self._expand(input_ids) if self.cfg.is_v4 else self.embed_tokens(input_ids)
        for layer, cache in zip(self.layers, kv_caches):
            x = layer(x, positions, cache)
        if self.cfg.is_v4:
            x = self.hc_head(x)
        return self.norm(x)

    def decode(self, input_ids: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
               context_lens: torch.Tensor, slot_mapping: torch.Tensor, kv_caches: list[torch.Tensor],
               ngram_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = self._expand(input_ids) if self.cfg.is_v4 else self.embed_tokens(input_ids)
        for layer, cache in zip(self.layers, kv_caches):
            x = layer.decode(x, positions, block_tables, context_lens, slot_mapping, ngram_ids, cache) \
                if self.cfg.is_v4 else layer.decode(x, positions, block_tables, context_lens, slot_mapping, cache)
        if self.cfg.is_v4:
            x = self.hc_head(x)
        return self.norm(x)

    def prefill_batched(self, input_ids: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
                        cached_lens: torch.Tensor, new_lens: torch.Tensor, slot_mapping: torch.Tensor,
                        kv_caches: list[torch.Tensor], ngram_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = self._expand(input_ids) if self.cfg.is_v4 else self.embed_tokens(input_ids)
        for layer, cache in zip(self.layers, kv_caches):
            x = layer.prefill_batched(x, positions, block_tables, cached_lens, new_lens, slot_mapping, ngram_ids, cache) \
                if self.cfg.is_v4 else layer.prefill_batched(x, positions, block_tables, cached_lens, new_lens, slot_mapping, cache)
        if self.cfg.is_v4:
            x = self.hc_head(x)
        return self.norm(x)


class DeepseekForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, max_len: int):
        super().__init__()
        self.config = cfg
        self.model = DeepseekModel(cfg, max_len)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: list[torch.Tensor],
                last_token_mask: torch.Tensor | None = None, ngram_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = self.model(input_ids, positions, kv_caches, ngram_ids)
        if last_token_mask is not None:  # 分块 prefill 时只有收尾的序列需要 logits
            x = x[last_token_mask]
        return self.lm_head(x)

    def forward_prefill_batched(self, input_ids: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
                                cached_lens: torch.Tensor, new_lens: torch.Tensor, slot_mapping: torch.Tensor,
                                kv_caches: list[torch.Tensor], ngram_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = self.model.prefill_batched(input_ids, positions, block_tables, cached_lens, new_lens, slot_mapping, kv_caches, ngram_ids)
        last = x[torch.arange(x.size(0), device=x.device), new_lens - 1]
        return self.lm_head(last)

    def forward_decode(self, input_ids: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
                       context_lens: torch.Tensor, slot_mapping: torch.Tensor,
                       kv_caches: list[torch.Tensor], ngram_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.model.decode(input_ids, positions, block_tables, context_lens, slot_mapping, kv_caches, ngram_ids))
