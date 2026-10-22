import torch
from torch import nn

from nanodeepinfer.config import ModelConfig
from nanodeepinfer.layers.layernorm import RMSNorm
from nanodeepinfer.layers.rotary_embedding import RotaryEmbedding
from nanodeepinfer.utils.context import get_context


class V4SparseAttention(nn.Module):
    """V4 稀疏 MLA：索引器给历史潜向量打分选 topk，主注意力只与选中的潜向量计算。

    对应原版 v4_modules/{attn,index_attn,mla_attn}.py 的主线，三点关键差异于 V3：
    1. head_dim = kv_lora_rank + qk_rope_head_dim，q 直接与缓存潜向量内积（FlashMLA 布局），
       RoPE 只加在尾部 qk_rope_head_dim 维；
    2. 稀疏索引器（MQAIndexer）：index q 由 q_B 输出尾部分裂、index k 从主潜向量经
       k_proj 投影（use_kproj 变体，不另设缓存），打分 logits = Σ_h relu(iq_h·ik_l)·w_h，
       每个 query 选 index_topk 个位置，其余在 mask 中屏蔽；
    3. V 侧低秩：加权潜向量按 o_groups 分组过 Wv，再经 out_proj 回到 hidden。
    原版 indexer Q/K 与稀疏 KV 的 FP4 量化、跨层共享属内核优化，此处不复制。
    """

    def __init__(self, cfg: ModelConfig, max_len: int):
        super().__init__()
        assert cfg.q_lora_rank is not None and cfg.index_topk > 0
        self.num_heads = cfg.num_attention_heads
        self.kv_rank = cfg.kv_lora_rank
        self.rope_dim = cfg.qk_rope_head_dim
        self.head_dim = self.kv_rank + self.rope_dim  # V4: q 与潜向量同维
        self.scale = self.head_dim ** -0.5
        self.idx_heads = cfg.index_n_heads
        self.idx_dim = cfg.index_head_dim
        self.topk = cfg.index_topk
        self.o_groups = cfg.o_groups or 1
        self.o_head_dim = cfg.o_lora_rank or self.head_dim
        assert self.num_heads % self.o_groups == 0
        self.window = cfg.window_size
        self.wq_a = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.q_b = nn.Linear(cfg.q_lora_rank, self.num_heads * self.head_dim + self.idx_heads * self.idx_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps) if cfg.use_qnorm else None
        self.wkv_a = nn.Linear(cfg.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        # 索引器：k 从主潜向量投影；w 给出各头打分权重
        self.index_k_proj = nn.Linear(self.head_dim, self.idx_dim, bias=False)
        self.index_k_norm = RMSNorm(self.idx_dim, cfg.rms_norm_eps) if cfg.index_use_knorm else None
        self.index_w_proj = nn.Linear(cfg.hidden_size, self.idx_heads, bias=False)
        # V 侧：一组头 (n_head/o_groups)*head_dim -> o_head_dim，再汇总回 hidden
        self.wv = nn.Linear((self.num_heads // self.o_groups) * self.head_dim, self.o_groups * self.o_head_dim, bias=False)
        self.out_proj = nn.Linear(self.o_groups * self.o_head_dim, cfg.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.rope_dim, max_len, cfg.rope_theta)
        self.sink = (nn.Parameter(torch.full((self.num_heads,), cfg.attn_sink_init_value))
                     if cfg.attn_sink_init_value is not None else None)

    def _projections(self, x: torch.Tensor, positions: torch.Tensor):
        """q（含 rope）、index q/w、待写缓存的潜向量（含尾部 rope）。"""
        q = self.q_b(self.q_a_norm(self.wq_a(x)))
        q, iq = q.split([self.num_heads * self.head_dim, self.idx_heads * self.idx_dim], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = torch.cat([q[..., : -self.rope_dim], self.rotary_emb(q[..., -self.rope_dim:], positions)], dim=-1)
        iq = iq.view(-1, self.idx_heads, self.idx_dim)
        iq = torch.cat([iq[..., : -self.rope_dim], self.rotary_emb(iq[..., -self.rope_dim:], positions)], dim=-1)
        latent = self.kv_norm(self.wkv_a(x))
        latent = torch.cat([latent[..., : -self.rope_dim], self.rotary_emb(latent[..., -self.rope_dim:], positions).squeeze(1)], dim=-1)
        iw = self.index_w_proj(x)
        return q, iq, iw, latent

    def _index_k(self, latent_all: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """整段缓存潜向量 -> index k（尾部 rope，按各位置的 positions）。"""
        ik = self.index_k_proj(latent_all)
        if self.index_k_norm is not None:
            ik = self.index_k_norm(ik)
        lead = ik.shape[:-1]
        tail = self.rotary_emb(ik[..., -self.rope_dim:].reshape(-1, 1, self.rope_dim),
                               positions.reshape(-1)).view(*lead, self.rope_dim)
        return torch.cat([ik[..., : -self.rope_dim], tail], dim=-1)

    def _sparse_mask(self, iq: torch.Tensor, iw: torch.Tensor, ik: torch.Tensor,
                     valid: torch.Tensor) -> torch.Tensor:
        """relu(iq·ik)·w 头加权打分 -> topk 布尔掩码（True=选中）。valid 为缓存有效位。"""
        if ik.dim() == 3:  # 整批：iq [b,(s,)H,D]，ik [b,L,D]
            eq = "bshd,bld->bshl" if iq.dim() == 4 else "bhd,bld->bhl"
        else:              # 单序列：iq [s,H,D]，ik [L,D]
            eq = "shd,ld->shl"
        logits = torch.einsum(eq, iq, ik).float().relu_()
        logits = torch.einsum("...hl,...h->...l", logits, iw.float())
        if valid.dim() > 1 and valid.dim() < logits.dim():
            valid = valid.unsqueeze(-2)  # [b,L] -> [b,1,L]，对齐 batched prefill 的 [b,s,L]
        logits = logits.masked_fill(~valid, float("-inf"))
        L = logits.size(-1)
        k = min(self.topk, L)
        picked = logits.topk(k, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, picked, True)
        return mask & valid

    def _attend(self, q: torch.Tensor, latent: torch.Tensor, mask: torch.Tensor | None,
                wv: torch.Tensor) -> torch.Tensor:
        if q.dim() == 4:    # 整批 prefill: q [B,S,H,D]，缓存 [B,L,D]
            score_eq, out_eq = "bshd,bld->bhsl", "bhsl,bld->bshd"
        elif latent.dim() == 2:  # 单序列 prefill: q [S,H,D]，缓存 [L,D]
            score_eq, out_eq = "shd,ld->hsl", "hsl,ld->shd"
        else:               # 整批 decode: q [B,H,D]，缓存 [B,L,D]
            score_eq, out_eq = "bhd,bld->bhl", "bhl,bld->bhd"
        scores = torch.einsum(score_eq, q, latent) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        if self.sink is not None:  # sink 作为额外一列参与 softmax，其"值"为零
            sink = self.sink.view(*([1] * (scores.dim() - 2)), self.num_heads, 1).expand_as(scores[..., :1])
            scores = torch.cat([scores, sink.to(scores.dtype)], dim=-1)
        attn = torch.softmax(scores.float(), dim=-1).to(latent.dtype)
        if self.sink is not None:
            attn = attn[..., :-1]
        out = torch.einsum(out_eq, attn, latent)  # [..., H, D]（加权潜向量）
        g = self.o_groups
        new_shape = out.shape[:-2] + (g, (self.num_heads // g) * self.head_dim)
        return torch.nn.functional.linear(out.reshape(new_shape), wv)

    def _causal_window_mask(self, q_pos: torch.Tensor, kv_pos: torch.Tensor) -> torch.Tensor:
        mask = kv_pos.unsqueeze(-2) > q_pos.unsqueeze(-1)
        if self.window > 0:
            mask |= (q_pos.unsqueeze(-1) - kv_pos.unsqueeze(-2)) >= self.window
        return mask

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        q, iq, iw, latent = self._projections(x, positions)
        flat = kv_cache.view(-1, self.head_dim)
        flat[ctx.slot_mapping] = latent
        out = x.new_empty((x.size(0), self.o_groups * self.o_head_dim))
        offset, block_size = 0, kv_cache.size(1)
        for meta in ctx.seq_metas:
            s, total = meta.scheduled_len, meta.total_len
            pos = torch.arange(total, device=x.device)
            slots = torch.tensor(meta.block_table, device=x.device)[pos // block_size] * block_size + pos % block_size
            lat_all = flat[slots]
            q_pos = torch.arange(meta.cached_len, total, device=x.device)
            causal = self._causal_window_mask(q_pos, pos)
            ik = self._index_k(lat_all, pos)
            sparse = self._sparse_mask(iq[offset: offset + s], iw[offset: offset + s], ik, pos >= 0)
            out[offset: offset + s] = self._attend(
                q[offset: offset + s], lat_all, causal | ~sparse, self.wv.weight).reshape(s, -1)
            offset += s
        return self.out_proj(out)

    def prefill_batched(self, x: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
                        cached_lens: torch.Tensor, new_lens: torch.Tensor, slot_mapping: torch.Tensor,
                        kv_cache: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q, iq, iw, latent = self._projections(x.reshape(b * s, -1), positions.reshape(-1))
        flat = kv_cache.view(-1, self.head_dim)
        flat[slot_mapping.reshape(-1)] = latent
        block_size = kv_cache.size(1)
        total_lens = cached_lens + new_lens
        length = int(total_lens.max())
        pos = torch.arange(length, device=x.device)
        slots = torch.gather(block_tables, 1, (pos // block_size).unsqueeze(0).expand(b, -1)) * block_size + (pos % block_size).unsqueeze(0)
        lat_all = flat[slots]
        valid = pos.unsqueeze(0) < total_lens.unsqueeze(1)
        q_pos = cached_lens.unsqueeze(1) + torch.arange(s, device=x.device).unsqueeze(0)
        q_pos = torch.minimum(q_pos, (total_lens - 1).unsqueeze(1))
        causal = pos.view(1, 1, length) > q_pos.view(b, s, 1)
        if self.window > 0:
            causal |= (q_pos.view(b, s, 1) - pos.view(1, 1, length)) >= self.window
        causal |= ~valid.view(b, 1, length)
        ik = self._index_k(lat_all, pos.unsqueeze(0).expand(b, -1))
        sparse = self._sparse_mask(iq.view(b, s, self.idx_heads, -1), iw.view(b, s, -1), ik, valid)
        out = self._attend(q.view(b, s, self.num_heads, -1), lat_all,
                           causal.unsqueeze(1) | ~sparse.unsqueeze(1), self.wv.weight)
        return self.out_proj(out.reshape(b, s, -1))

    def decode(self, x: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
               context_lens: torch.Tensor, slot_mapping: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        q, iq, iw, latent = self._projections(x, positions)
        flat = kv_cache.view(-1, self.head_dim)
        flat[slot_mapping] = latent
        block_size = kv_cache.size(1)
        length = block_tables.size(1) * block_size
        pos = torch.arange(length, device=x.device)
        slots = torch.gather(block_tables, 1, (pos // block_size).unsqueeze(0).expand(x.size(0), -1)) * block_size + (pos % block_size).unsqueeze(0)
        lat_all = flat[slots]
        valid = pos.unsqueeze(0) < context_lens.unsqueeze(1)
        ik = self._index_k(lat_all, pos.unsqueeze(0).expand(x.size(0), -1))
        sparse = self._sparse_mask(iq, iw, ik, valid)
        mask = ~valid.unsqueeze(1) | ~sparse.unsqueeze(1)
        if self.window > 0:
            mask |= ((context_lens - 1).view(-1, 1, 1) - pos.view(1, 1, length)) >= self.window
        out = self._attend(q, lat_all, mask, self.wv.weight)
        return self.out_proj(out.reshape(x.size(0), -1))
