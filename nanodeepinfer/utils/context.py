from dataclasses import dataclass, field

import torch


@dataclass
class SeqMeta:
    block_table: list[int]
    cached_len: int  # 本步之前已写入 KV cache 的 token 数
    scheduled_len: int  # 本步要计算的 token 数

    @property
    def total_len(self) -> int:
        return self.cached_len + self.scheduled_len


@dataclass
class Context:
    is_prefill: bool
    seq_metas: list[SeqMeta] = field(default_factory=list)
    slot_mapping: torch.Tensor | None = None  # 每个待计算 token 在分页 KV cache 中的物理槽位


_context: Context | None = None


def set_context(ctx: Context):
    global _context
    _context = ctx


def get_context() -> Context:
    assert _context is not None
    return _context


def reset_context():
    global _context
    _context = None
