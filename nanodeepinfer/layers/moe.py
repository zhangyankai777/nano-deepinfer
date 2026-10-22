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

    原版用 DeepEP 做跨卡专家通信、EPLB 做专家冗余重排；单机版保留
    分组受限 topk 的选择逻辑本身，专家计算在本地完成。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if cfg.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.zeros(cfg.n_routed_experts))
        self.top_k = cfg.num_experts_per_tok
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.scoring = cfg.scoring_func
        self.norm_topk_prob = cfg.norm_topk_prob
        self.scaling = cfg.routed_scaling_factor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = (x @ self.weight.t()).float()
        scores = torch.sigmoid(logits) if self.scoring == "sigmoid" else torch.softmax(logits, dim=-1)
        choice = scores + self.e_score_correction_bias if hasattr(self, "e_score_correction_bias") else scores
        if self.n_group > 1:
            per_group = choice.size(-1) // self.n_group
            g = choice.view(-1, self.n_group, per_group)
            group_scores = g.topk(2, dim=-1).values.sum(-1)
            group_idx = group_scores.topk(self.topk_group, dim=-1).indices
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(-1, -1, per_group).reshape(x.shape[:-1] + (-1,))
            choice = choice.masked_fill(score_mask == 0, float("-inf"))
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
        self._stacked: tuple[torch.Tensor, torch.Tensor] | None = None

    def _stacked_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._stacked is None:  # 权重加载完成后一次性堆叠，之后地址不变，可被 CUDA graph 捕获
            gate_up = torch.stack([torch.cat([e.gate_proj.weight, e.up_proj.weight]) for e in self.experts])
            down = torch.stack([e.down_proj.weight for e in self.experts])
            self._stacked = (gate_up, down)
        return self._stacked

    def _routed(self, flat: torch.Tensor, idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """全部专家堆叠成批量矩阵乘：无逐专家循环、无动态形状，decode 可被 CUDA graph 捕获。

        代价是每个 token 对所有专家都算一遍（靠路由权重把未选专家置零），
        换来的是大 GEMM 与零 CPU 同步；大批量时按行分块控制中间张量大小。
        """
        gate_up, down = self._stacked_weights()
        expert_w = torch.zeros(flat.size(0), gate_up.size(0), dtype=weights.dtype, device=flat.device)
        expert_w.scatter_add_(1, idx, weights)
        outs = []
        for chunk in flat.split(1024):
            w = expert_w[: chunk.size(0)]
            expert_w = expert_w[chunk.size(0):]
            h = torch.einsum("nh,eih->eni", chunk, gate_up)
            gate, up = h.chunk(2, dim=-1)
            h = torch.nn.functional.silu(gate) * up
            out_e = torch.einsum("eni,ehi->enh", h, down)
            outs.append(torch.einsum("enh,ne->nh", out_e, w.to(h.dtype)))
        return torch.cat(outs) if len(outs) > 1 else outs[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(-1, x.size(-1))
        idx, weights = self.gate(flat)
        out = self._routed(flat, idx, weights)
        if hasattr(self, "shared_experts"):
            out = out + self.shared_experts(flat)
        return out.view_as(x)
