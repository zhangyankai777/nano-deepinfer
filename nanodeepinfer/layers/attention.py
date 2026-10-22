import torch
from torch import nn

from nanodeepinfer.config import ModelConfig
from nanodeepinfer.layers.layernorm import RMSNorm
from nanodeepinfer.layers.rotary_embedding import RotaryEmbedding
from nanodeepinfer.utils.context import get_context


class MLAAttention(nn.Module):
    """Multi-head Latent Attention：KV cache 只存低秩潜向量 c_kv 与解耦 RoPE 键。

    计算采用吸收式（absorbed）形式：把 kv_b_proj 中生成 K 的部分预先吸收进 Q、
    生成 V 的部分留到注意力加权之后再作用于潜向量，于是每步只需与
    kv_lora_rank 维的潜缓存做注意力，不必把全部历史 KV 上投影回各头空间。
    这正是 FlashMLA 在内核里采用的等价变换。
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

    def _projections(self, x: torch.Tensor, positions: torch.Tensor):
        """返回吸收后的 q_abs、带 RoPE 的 q_pe，以及待写入缓存的潜向量。"""
        if hasattr(self, "q_proj"):
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(-1, self.num_heads, self.qk_dim)
        kv = self.kv_a_proj_with_mqa(x)
        c_kv = self.kv_a_layernorm(kv[..., : self.kv_rank])
        k_pe = self.rotary_emb(kv[..., self.kv_rank:].unsqueeze(1), positions).squeeze(1)
        w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope + self.v_dim, self.kv_rank)
        q_abs = torch.einsum("nha,har->nhr", q[..., : self.qk_nope], w[:, : self.qk_nope])
        q_pe = self.rotary_emb(q[..., self.qk_nope:], positions)
        return q_abs, q_pe, c_kv, k_pe, w[:, self.qk_nope:].transpose(1, 2)  # [H, kv_rank, v_dim]

    def _attend(self, q_abs, q_pe, w_v, latent, mask):
        # q 两部分拼回与缓存相同的布局后只需一次 GEMM：前段与潜向量、后段与 RoPE 键分别内积
        q_full = torch.cat([q_abs, q_pe], dim=-1)
        if q_full.dim() == 4:  # 整批 prefill: q [B,S,H,D]，缓存 [B,L,D]
            score_eq, out_eq, proj_eq = "bshd,bld->bhsl", "bhsl,blr->bshr", "bshr,hrv->bshv"
        elif latent.dim() == 2:  # 单序列 prefill: q [S,H,D]，缓存 [L,D]
            score_eq, out_eq, proj_eq = "shd,ld->hsl", "hsl,lr->shr", "nhr,hrv->nhv"
        else:  # 整批 decode: q [B,H,D]，缓存 [B,L,D]
            score_eq, out_eq, proj_eq = "bhd,bld->bhl", "bhl,blr->bhr", "nhr,hrv->nhv"
        scores = torch.einsum(score_eq, q_full, latent) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(latent.dtype)
        out_lat = torch.einsum(out_eq, attn, latent[..., : self.kv_rank])
        return torch.einsum(proj_eq, out_lat, w_v)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        q_abs, q_pe, c_kv, k_pe, w_v = self._projections(x, positions)
        flat = kv_cache.view(-1, self.kv_rank + self.qk_rope)
        flat[ctx.slot_mapping] = torch.cat([c_kv, k_pe], dim=-1)

        out = x.new_empty((x.size(0), self.num_heads, self.v_dim))
        offset, block_size = 0, kv_cache.size(1)
        for meta in ctx.seq_metas:
            s, total = meta.scheduled_len, meta.total_len
            pos = torch.arange(total, device=x.device)
            slots = torch.tensor(meta.block_table, device=x.device)[pos // block_size] * block_size + pos % block_size
            latent = flat[slots]
            q_pos = torch.arange(meta.cached_len, total, device=x.device)
            mask = pos.unsqueeze(0) > q_pos.view(-1, 1)
            out[offset: offset + s] = self._attend(
                q_abs[offset: offset + s], q_pe[offset: offset + s], w_v, latent, mask)
            offset += s
        return self.o_proj(out.reshape(x.size(0), -1))

    def prefill_batched(self, x: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
                        cached_lens: torch.Tensor, new_lens: torch.Tensor, slot_mapping: torch.Tensor,
                        kv_cache: torch.Tensor) -> torch.Tensor:
        """整批 prefill：x [B,S,H]，填充行指向草稿槽位且其输出被丢弃。"""
        b, s, _ = x.shape
        q_abs, q_pe, c_kv, k_pe, w_v = self._projections(x.reshape(b * s, -1), positions.reshape(-1))
        flat = kv_cache.view(-1, self.kv_rank + self.qk_rope)
        flat[slot_mapping.reshape(-1)] = torch.cat([c_kv, k_pe], dim=-1)
        block_size = kv_cache.size(1)
        total_lens = cached_lens + new_lens
        length = int(total_lens.max())
        pos = torch.arange(length, device=x.device)
        block_idx = (pos // block_size).unsqueeze(0).expand(b, -1)
        slots = torch.gather(block_tables, 1, block_idx) * block_size + (pos % block_size).unsqueeze(0)
        latent = flat[slots]
        q_pos = cached_lens.unsqueeze(1) + torch.arange(s, device=x.device).unsqueeze(0)
        q_pos = torch.minimum(q_pos, (total_lens - 1).unsqueeze(1))
        mask = (pos.view(1, 1, 1, -1) > q_pos.view(b, 1, s, 1)) | (pos.view(1, 1, 1, -1) >= total_lens.view(b, 1, 1, 1))
        out = self._attend(q_abs.view(b, s, self.num_heads, -1), q_pe.view(b, s, self.num_heads, -1), w_v, latent, mask)
        return self.o_proj(out.reshape(b, s, -1))

    def decode(self, x: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
               context_lens: torch.Tensor, slot_mapping: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        """整批 decode：形状全部静态，可被 CUDA graph 捕获。"""
        q_abs, q_pe, c_kv, k_pe, w_v = self._projections(x, positions)
        flat = kv_cache.view(-1, self.kv_rank + self.qk_rope)
        flat[slot_mapping] = torch.cat([c_kv, k_pe], dim=-1)
        block_size = kv_cache.size(1)
        length = block_tables.size(1) * block_size
        pos = torch.arange(length, device=x.device)
        block_idx = (pos // block_size).unsqueeze(0).expand(x.size(0), -1)
        slots = torch.gather(block_tables, 1, block_idx) * block_size + (pos % block_size).unsqueeze(0)
        latent = flat[slots]
        mask = (pos.unsqueeze(0) >= context_lens.unsqueeze(1)).unsqueeze(1)
        out = self._attend(q_abs, q_pe, w_v, latent, mask)
        return self.o_proj(out.reshape(x.size(0), -1))
