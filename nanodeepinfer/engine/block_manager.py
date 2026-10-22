import hashlib
from collections import deque

from nanodeepinfer.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """分页 KV cache 分配 + 前缀缓存：整块按 (父块哈希, 本块 token) 链式哈希复用。"""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(prefix.to_bytes(8, "little", signed=True))
        h.update(b"".join(t.to_bytes(4, "little") for t in token_ids))
        return int.from_bytes(h.digest(), "little", signed=True)

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def can_allocate(self, seq: Sequence) -> int:
        """返回可命中的前缀块数，显存不足返回 -1。"""
        h, num_cached = -1, 0
        num_new = seq.num_blocks
        for i in range(seq.num_blocks - 1):  # 最后一块未写满，不参与复用
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached += 1
            if block_id in self.used_block_ids:
                num_new -= 1
        return num_cached if len(self.free_block_ids) >= num_new else -1

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        h = -1
        for i in range(num_cached_blocks):
            h = self.compute_hash(seq.block(i), h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.used_block_ids.remove(block_id)
                self.free_block_ids.append(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            h = self.compute_hash(seq.block(i), h)
            block.hash, block.token_ids = h, seq.block(i)
            self.hash_to_block_id[h] = block.block_id
