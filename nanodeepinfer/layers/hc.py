import torch
from torch import nn

from nanodeepinfer.config import ModelConfig


def sinkhorn(x: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """双随机归一化：保持多路残差混合时信息守恒（对应原版 mhc_sinkhorn）。"""
    for _ in range(iters):
        x = x / x.sum(-1, keepdim=True).clamp_min(eps)
        x = x / x.sum(-2, keepdim=True).clamp_min(eps)
    return x


class HCMix(nn.Module):
    """一个子层（attn 或 ffn）的 mHC 参数：fn 投影 + 三组 sigmoid 的 scale/base。

    base 的初始化沿用原版约定：pre 段除本层对齐的一路为 +pre 外均取 -pre，
    comb 段取 (2I-1)*res——即初始时残差近似恒等传递，mix 交给 fn 学习。
    """

    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        m, h = cfg.hc_mult, cfg.hidden_size
        self.hc_mult = m
        pre, post, res = cfg.hc_pre_init_value, cfg.hc_post_init_value, cfg.hc_res_init_value
        base = torch.cat([
            torch.full((m,), -pre),
            torch.full((m,), post),
            (torch.eye(m) * 2 * res - res).flatten(),
        ])
        base[layer_id % m] = pre
        self.base = nn.Parameter(base)
        self.fn = nn.Parameter(torch.empty((2 + m) * m, m * h))
        nn.init.zeros_(self.fn)
        self.scale = nn.Parameter(torch.full((3,), cfg.hc_dynamic_scale))
        self.eps = cfg.hc_eps
        self.post_mult = cfg.hc_post_mult_value
        self.sinkhorn_iters = cfg.hc_sinkhorn_iters

    def mixes(self, trunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """trunk [...,m,h] -> pre_mix [...,m], post_mix [...,m], comb_mix [...,m,m]。"""
        m = self.hc_mult
        x = trunk.flatten(-2).float()  # [..., m*h]，兼容 [n,m,h] 与整批 prefill 的 [B,S,m,h]
        rms = x.square().mean(-1, keepdim=True).add(self.eps).rsqrt()  # PolyNorm：fn 已折叠 norm 权重
        raw = (x @ self.fn.float().t()) * rms
        pre_raw, post_raw, comb_raw = raw.split([m, m, m * m], dim=-1)
        pre_mix = torch.sigmoid(pre_raw * self.scale[0] + self.base[:m]) + self.eps
        post_mix = torch.sigmoid(post_raw * self.scale[1] + self.base[m: 2 * m]) * self.post_mult
        comb = torch.sigmoid(comb_raw * self.scale[2] + self.base[2 * m:]).unflatten(-1, (m, m))
        return pre_mix, post_mix, sinkhorn(comb, self.sinkhorn_iters, self.eps)

    def pre(self, trunk: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """子层前：把多路 trunk 混合成单路输入，同时留下 post 所需的混合系数。"""
        pre_mix, post_mix, comb = self.mixes(trunk)
        layer_in = torch.einsum("...m,...mh->...h", pre_mix.to(trunk.dtype), trunk)
        return layer_in, (post_mix, comb)

    @staticmethod
    def post(y: torch.Tensor, trunk: torch.Tensor, cached: tuple) -> torch.Tensor:
        """子层后：输出 y 按 post_mix 注入，trunk 按 comb 混合，得到新 trunk。"""
        post_mix, comb = cached
        mixed = torch.einsum("...ij,...jh->...ih", comb.to(trunk.dtype), trunk)
        return mixed + post_mix.to(y.dtype).unsqueeze(-1) * y.unsqueeze(-2)


class HCHead(nn.Module):
    """收尾：lm_head 之前把多路 trunk 混合回单路 hidden。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        m, h = cfg.hc_mult, cfg.hidden_size
        base = torch.full((m,), -cfg.hc_pre_init_value)
        base[0] = cfg.hc_pre_init_value
        self.base = nn.Parameter(base)
        self.fn = nn.Parameter(torch.empty(m, m * h))
        nn.init.zeros_(self.fn)
        self.scale = nn.Parameter(torch.tensor(cfg.hc_dynamic_scale))
        self.eps = cfg.hc_eps

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        x = trunk.flatten(-2).float()
        rms = x.square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        raw = (x @ self.fn.float().t()) * rms * self.scale + self.base
        mix = torch.sigmoid(raw) + self.eps
        return torch.einsum("...m,...mh->...h", mix.to(trunk.dtype), trunk)
