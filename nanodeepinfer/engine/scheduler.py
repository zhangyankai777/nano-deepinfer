from collections import deque

from nanodeepinfer.config import Config
from nanodeepinfer.engine.block_manager import BlockManager
from nanodeepinfer.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """连续批处理：优先 prefill（含分块），无新请求时整批 decode；显存不足按重算抢占。

    对应原版异步调度器的同步精简版——原版让 CPU 调度与设备执行重叠，
    这里保留同一套准入/抢占决策，只去掉零开销重叠的那层异步。
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled, num_batched = [], 0
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached = self.block_manager.can_allocate(seq)
                if num_cached == -1:
                    break
                self.block_manager.allocate(seq, num_cached)
            num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled:
                break
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            num_batched += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled.append(seq)
        if scheduled:
            return scheduled, True

        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        assert scheduled, "no sequence scheduled: KV cache too small for a single sequence"
        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int | None], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if token_id is None:  # 分块 prefill 尚未收尾
                continue
            seq.append_token(token_id)
            finished = token_id in seq.stop_token_ids or seq.num_completion_tokens == seq.max_tokens
            if not seq.ignore_eos and token_id == self.eos:
                finished = True
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
