import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """DeepSeek 的交错式 RoPE：相邻两维 (x0, x1) 构成一对共用一个频率旋转。

    缓冲用 cos/sin 两个实数张量保存（保证 Module.to(dtype) 不丢分量），
    前向时合成复数做一次乘法完成旋转。
    """

    def __init__(self, dim: int, max_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cis = torch.complex(self.cos_cache[positions].float(), self.sin_cache[positions].float())
        if x.dim() == 3:
            cis = cis.unsqueeze(1)
        pairs = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        return torch.view_as_real(pairs * cis).flatten(-2).to(x.dtype)
