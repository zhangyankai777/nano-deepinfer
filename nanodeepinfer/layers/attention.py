import torch
from torch import nn

from nanodeepinfer.config import ModelConfig
from nanodeepinfer.layers.layernorm import RMSNorm
from nanodeepinfer.layers.rotary_embedding import RotaryEmbedding
from nanodeepinfer.utils.context import get_context


class MLAAttention(nn.Module):
    """Multi-head Latent Attention：KV cache 只存低秩潜向量 c_kv 与解耦 RoPE 键。

    对应原版 FlashMLA 的数学等价路径——缓存 d_c=kv_lora_rank 维而非每头完整的 K/V，
    用到时再经 kv_b_proj 上投影还原各头 K/V，这是 DeepSeek 省显存的核心。
    """

    def __init__(self, cfg: ModelConfig, max_len: int):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.qk_nope = cfg.qk_nope_head_dim
        self.qk_rope = cfg.qk_rope_head_dim
        self.qk_dim = cfg.qk_head_dim
        self.v_dim = cfg.v_head_dim
        self.kv_rank = cfg.kv_lora_rank
        self.scale = self.qk_dim ** -0.5
        bias = cfg.attention_bias
        if cfg.q_lora_rank is None:
            self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.qk_dim, bias=bias)
        else:
            self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=bias)
            self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
            self.q_b_proj = nn.Linear(cfg.q_lora_rank, self.num_heads * self.qk_dim, bias=bias)
        self.kv_a_proj_with_mqa = nn.Linear(cfg.hidden_size, self.kv_rank + self.qk_rope, bias=bias)
        self.kv_a_layernorm = RMSNorm(self.kv_rank, cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_rank, self.num_heads * (self.qk_nope + self.v_dim), bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.v_dim, cfg.hidden_size, bias=bias)
        self.rotary_emb = RotaryEmbedding(self.qk_rope, max_len, cfg.rope_theta)

    def _q(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "q_proj"):
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(-1, self.num_heads, self.qk_dim)
        q_pe = self.rotary_emb(q[..., self.qk_nope:], positions)
        return torch.cat([q[..., : self.qk_nope], q_pe], dim=-1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        q = self._q(x, positions)
        kv = self.kv_a_proj_with_mqa(x)
        c_kv = self.kv_a_layernorm(kv[..., : self.kv_rank])
        k_pe = self.rotary_emb(kv[..., self.kv_rank:].unsqueeze(1), positions).squeeze(1)
        # 写入分页缓存：物理槽位由 ModelRunner 按 block_table 预先算好
        flat = kv_cache.view(-1, self.kv_rank + self.qk_rope)
        flat[ctx.slot_mapping] = torch.cat([c_kv, k_pe], dim=-1)

        out = x.new_empty((x.size(0), self.num_heads, self.v_dim))
        offset = 0
        block_size = kv_cache.size(1)
        for meta in ctx.seq_metas:
            s, total = meta.scheduled_len, meta.total_len
            pos = torch.arange(total, device=x.device)
            slots = torch.tensor(meta.block_table, device=x.device)[pos // block_size] * block_size + pos % block_size
            latent = flat[slots]
            kv_up = self.kv_b_proj(latent[..., : self.kv_rank]).view(total, self.num_heads, self.qk_nope + self.v_dim)
            k = torch.cat([kv_up[..., : self.qk_nope], latent[..., self.kv_rank:].unsqueeze(1).expand(-1, self.num_heads, -1)], dim=-1)
            v = kv_up[..., self.qk_nope:]
            q_s = q[offset: offset + s]
            scores = torch.einsum("shd,lhd->hsl", q_s.float(), k.float()) * self.scale
            q_pos = torch.arange(meta.cached_len, total, device=x.device)
            scores = scores.masked_fill(pos.unsqueeze(0) > q_pos.view(-1, 1), float("-inf"))
            attn = torch.softmax(scores, dim=-1).to(v.dtype)
            out[offset: offset + s] = torch.einsum("hsl,lhd->shd", attn, v)
            offset += s
        return self.o_proj(out.reshape(x.size(0), -1))
