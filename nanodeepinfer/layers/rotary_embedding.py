import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """DeepSeek 的交错式 RoPE：相邻两维 (x0, x1) 构成一对共用一个频率旋转。"""

    def __init__(self, dim: int, max_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[positions].to(x.dtype)
        sin = self.sin_cache[positions].to(x.dtype)
        if x.dim() == 3:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        pairs = x.float().unflatten(-1, (-1, 2))
        x0, x1 = pairs[..., 0], pairs[..., 1]
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return out.flatten(-2).to(x.dtype)
