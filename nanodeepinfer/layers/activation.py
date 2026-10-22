import torch
from torch import nn


class SwiGLU(nn.Module):
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * up
