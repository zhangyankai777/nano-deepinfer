import torch
from torch import nn

from nanodeepinfer.config import ModelConfig
from nanodeepinfer.layers.activation import SwiGLU


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(SwiGLU()(self.gate_proj(x), self.up_proj(x)))


class Gate(nn.Module):
    """V3 的无辅助损失路由：sigmoid 打分 + 校正偏置选专家，权重仍取原始打分。

    原版用 DeepEP 做跨卡专家通信、EPLB 做专家冗余重排；单机教学版保留
    分组受限 topk 的选择逻辑本身，专家计算退化为本地循环。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.hidden_size))
        if cfg.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.zeros(cfg.n_routed_experts))
        self.top_k = cfg.num_experts_per_tok
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.scoring = cfg.scoring_func
        self.norm_topk_prob = cfg.norm_topk_prob
        self.scaling = cfg.routed_scaling_factor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = x.float() @ self.weight.float().t()
        scores = torch.sigmoid(logits) if self.scoring == "sigmoid" else torch.softmax(logits, dim=-1)
        choice = scores + self.e_score_correction_bias if hasattr(self, "e_score_correction_bias") else scores
        if self.n_group > 1:
            g = choice.view(-1, self.n_group, choice.size(-1) // self.n_group)
            group_scores = g.topk(2, dim=-1).values.sum(-1)
            group_idx = group_scores.topk(self.topk_group, dim=-1).indices
            mask = torch.zeros_like(g).scatter_(1, group_idx.unsqueeze(-1), 1)
            choice = choice * mask.view(x.shape[:-1] + (-1,))
        topk_idx = choice.topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, topk_idx)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return topk_idx, (weights * self.scaling).to(x.dtype)


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = Gate(cfg)
        self.experts = nn.ModuleList([MLP(cfg.hidden_size, cfg.moe_intermediate_size) for _ in range(cfg.n_routed_experts)])
        if cfg.n_shared_experts:
            self.shared_experts = MLP(cfg.hidden_size, cfg.moe_intermediate_size * cfg.n_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(-1, x.size(-1))
        idx, weights = self.gate(flat)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token_idx, slot = (idx == e).nonzero(as_tuple=True)
            if token_idx.numel():
                out.index_add_(0, token_idx, expert(flat[token_idx]) * weights[token_idx, slot].unsqueeze(-1))
        if hasattr(self, "shared_experts"):
            out = out + self.shared_experts(flat)
        return out.view_as(x)
