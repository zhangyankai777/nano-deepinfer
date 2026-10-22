from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16
    ignore_eos: bool = False
    stop_token_ids: list[int] = field(default_factory=list)
    seq_seed: int | None = None  # 给定后同序列采样可复现，对应原版的确定性采样

    def __post_init__(self):
        assert self.temperature >= 0
        assert 0 < self.top_p <= 1
        assert self.top_k == -1 or self.top_k >= 1
        assert self.max_tokens >= 1

    @property
    def greedy(self) -> bool:
        return self.temperature < 1e-5
