from enum import Enum, auto
from itertools import count

from nanodeepinfer.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    _counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams, block_size: int):
        self.seq_id = next(self._counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = list(token_ids)
        self.last_token = token_ids[-1]
        self.num_prompt_tokens = len(token_ids)
        self.num_completion_tokens = 0
        self.block_size = block_size
        self.block_table: list[int] = []
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.stop_token_ids = sampling_params.stop_token_ids
        self.sampling_params = sampling_params

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        return (len(self) + self.block_size - 1) // self.block_size

    def block(self, i: int) -> list[int]:
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_completion_tokens += 1
