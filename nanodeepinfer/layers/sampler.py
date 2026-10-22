import torch

from nanodeepinfer.sampling_params import SamplingParams


class Sampler:
    """温度 / top-k / top-p 采样；seq_seed 给定后用独立发生器保证可复现。"""

    def forward(self, logits: torch.Tensor, params: list[SamplingParams]) -> list[int]:
        if all(p.greedy for p in params):  # 整批贪心只同步一次
            return logits.argmax(-1).tolist()
        tokens = [0] * len(params)
        greedy_rows = [i for i, p in enumerate(params) if p.greedy]
        if greedy_rows:
            picked = logits[greedy_rows].argmax(-1).tolist()
            for i, tok in zip(greedy_rows, picked):
                tokens[i] = tok
        for i, (row, p) in enumerate(zip(logits, params)):
            if p.greedy:
                continue
            probs = torch.softmax(row.float() / p.temperature, dim=-1)
            if p.top_k > 0:
                kth = probs.topk(p.top_k).values[-1]
                probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
            if p.top_p < 1.0:
                sorted_probs, order = probs.sort(descending=True)
                keep = sorted_probs.cumsum(-1) - sorted_probs < p.top_p
                probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs * keep)
            probs = probs / probs.sum()
            gen = None
            if p.seq_seed is not None:
                gen = torch.Generator(device=probs.device).manual_seed(p.seq_seed)
            tokens[i] = int(torch.multinomial(probs, 1, generator=gen))
        return tokens
