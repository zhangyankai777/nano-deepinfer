import torch
from torch import nn

from nanodeepinfer.config import ModelConfig
from nanodeepinfer.layers.attention import MLAAttention
from nanodeepinfer.layers.layernorm import RMSNorm
from nanodeepinfer.layers.moe import MLP, MoE


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


class DeepseekModel(nn.Module):
    def __init__(self, cfg: ModelConfig, max_len: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i, max_len) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: list[torch.Tensor]) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer, cache in zip(self.layers, kv_caches):
            x = layer(x, positions, cache)
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
                last_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.model(input_ids, positions, kv_caches)
        if last_token_mask is not None:  # 分块 prefill 时只有收尾的序列需要 logits
            x = x[last_token_mask]
        return self.lm_head(x)
