import numpy as np
import torch
from torch import nn

from nanodeepinfer.config import ModelConfig


def _next_prime(n: int, seen: set[int]) -> int:
    """原版用 sympy.nextprime 且全局去重，这里内联等价实现避免额外依赖。"""
    def is_prime(x: int) -> bool:
        return x > 1 and all(x % p for p in range(2, int(x ** 0.5) + 1))
    p = n + 1
    while not is_prime(p) or p in seen:
        p += 1
    seen.add(p)
    return p


class NgramHash:
    """n-gram 哈希元数据：每层一组随机奇数乘法因子，每阶每头一张素数大小的表。

    与原版 engram_metadata.NgramHashMapping 同约定：乘法因子由
    seed + 10007 * layer_id 的 NumPy 发生器导出；表大小取全局互不相同的素数。
    """

    def __init__(self, cfg: ModelConfig, device: torch.device):
        ecfg_vocab = cfg.engram_vocab_size or [cfg.vocab_size] * (cfg.engram_max_ngram_size - 1)
        num_heads = cfg.n_head_per_ngram[0]
        seen: set[int] = set()
        sizes = []
        for _layer in cfg.engram_layer_ids:
            per_ngram = []
            for v in ecfg_vocab:
                start = int(v) - 1
                per_ngram.append([_next_prime(start, seen) for _ in range(num_heads)])
            sizes.append(per_ngram)
        bound = max(1, (np.iinfo(np.int64).max // int(cfg.vocab_size)) // 2)
        mults = []
        for layer_id in cfg.engram_layer_ids:
            rng = np.random.default_rng(10007 * int(layer_id))
            mults.append(rng.integers(0, bound, size=(cfg.engram_max_ngram_size,)) * 2 + 1)
        self.multipliers = torch.tensor(np.stack(mults), dtype=torch.int64, device=device)
        self.table_sizes = torch.tensor(sizes, dtype=torch.int64, device=device)  # [L, K, H]
        self.offsets = torch.zeros(len(sizes), len(ecfg_vocab), num_heads + 1, dtype=torch.int64, device=device)
        self.offsets[..., 1:] = self.table_sizes.cumsum(-1)

    def to(self, device: torch.device) -> "NgramHash":
        self.multipliers = self.multipliers.to(device)
        self.table_sizes = self.table_sizes.to(device)
        self.offsets = self.offsets.to(device)
        return self

    def hashes(self, layer_slot: int, ngram_ids: torch.Tensor) -> torch.Tensor:
        """ngram_ids [n, max_n]（每位置以当前 token 结尾的窗口）-> [n, K*H] 表内行号。"""
        K = self.table_sizes.size(1)
        tok = ngram_ids.to(torch.int64)
        out = []
        for k in range(K):  # 阶 k+2：用末尾 k+2 个 token
            window = tok[:, tok.size(1) - (k + 2):]
            h = (window * self.multipliers[layer_slot, : k + 2]).sum(-1, keepdim=True)  # [n,1]
            h = h.remainder(self.table_sizes[layer_slot, k]) + self.offsets[layer_slot, k, :-1]
            out.append(h)
        return torch.cat(out, dim=-1)


class Engram(nn.Module):
    """n-gram 条件记忆：按哈希查稀疏嵌入表，投影后与 HC trunk 门控融合。

    对应原版 v4_modules/engram.py（DeepEP 跨卡取表简化为本地查表）；门控形式
    （hidden 与共享的第 m+1 路 kv 算 sigmoid 门、逐路注入 kv）按 tile_kernels
    engram_gate 的接口语义复刻。
    """

    def __init__(self, cfg: ModelConfig, layer_slot: int, hash_map: NgramHash):
        super().__init__()
        self.layer_slot = layer_slot
        self.hash_map = hash_map
        m = cfg.hc_mult
        n_embed = sum(cfg.n_embed_per_ngram)
        num_heads = cfg.n_head_per_ngram[0]
        head_dim = cfg.n_embed_per_ngram[0] // num_heads
        total_rows = int(hash_map.offsets[layer_slot, :, -1].sum())
        self.embedding = nn.Embedding(total_rows, head_dim)
        self.fused_kv_projection = nn.Linear(n_embed, cfg.hidden_size * (m + 1), bias=False)
        self.weight_hidden = nn.Parameter(torch.empty(m, cfg.hidden_size))
        self.weight_embed = nn.Parameter(torch.empty(m, cfg.hidden_size))
        nn.init.zeros_(self.weight_hidden)
        nn.init.zeros_(self.weight_embed)
        self.eps = cfg.engram_eps
        self.clamp = cfg.engram_clamp_value
        self.head_dim = head_dim
        self.k_heads = hash_map.table_sizes.size(1) * num_heads

    def forward(self, trunk: torch.Tensor, ngram_ids: torch.Tensor) -> torch.Tensor:
        """trunk [..., m, hidden]；ngram_ids [..., max_n] -> 注入记忆后的新 trunk。"""
        lead = trunk.shape[:-2]
        t3 = trunk.flatten(0, -3) if trunk.dim() > 3 else trunk
        ids = ngram_ids.reshape(-1, ngram_ids.size(-1))
        idx = self.hash_map.hashes(self.layer_slot, ids)             # [n, K*H]
        emb = self.embedding(idx).reshape(idx.size(0), -1)           # [n, K*H*head_dim]
        kv = self.fused_kv_projection(emb).unflatten(-1, (t3.size(1) + 1, -1)).to(t3.dtype)
        gate = torch.sigmoid(
            torch.einsum("mh,nmh->nm", self.weight_hidden.float(), t3.float())
            + torch.einsum("mh,nh->nm", self.weight_embed.float(), kv[:, -1].float())
            + self.eps)
        out = t3 + (gate.to(t3.dtype).unsqueeze(-1) * kv[:, :-1]).clamp(-self.clamp, self.clamp)
        return out.unflatten(0, lead) if trunk.dim() > 3 else out
