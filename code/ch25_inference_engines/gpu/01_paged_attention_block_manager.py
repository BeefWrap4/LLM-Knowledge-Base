# ---
# chapter: 25
# topic: PagedAttention Block Manager
# section: 25.2.1
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 01_paged_attention_block_manager.py
# expected_runtime: <1s
# expected_output: 演示 PagedAttention 的 logical→physical block table 映射、allocation/copy-on-write
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.2.1
# Interview hooks:
#   1. PagedAttention 为什么能提升显存利用率？(答: 类比 OS 虚拟内存，按页分配，碎片化低，复用率高)
#   2. Block size 如何选择？过小过大各有什么影响？(答: 16 token 经验值，过小页表大，过大内部碎片)
#   3. Beam Search 场景下如何处理 block 共享？(答: Copy-on-Write 写时复制)

"""PagedAttention Block Manager (educational mock).

A pedagogical implementation of the vLLM PagedAttention block manager.
Real vLLM uses a C++/CUDA block table; this Python version shows the
*logical* contract: sequences own a list of block IDs, blocks are a
global pool, and a sequence may share a prefix with another sequence
(via a reference count) without copying the underlying KV tensors.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Block:
    block_id: int
    ref_count: int = 0  # how many sequences currently point at this block
    is_full: bool = False


@dataclass
class BlockTable:
    """Per-sequence mapping: token_index → physical block id."""
    seq_id: int
    block_ids: list[int] = field(default_factory=list)

    def append_block(self, block_id: int) -> None:
        self.block_ids.append(block_id)


class BlockManager:
    """Mock of vLLM's BlockManager.

    In real vLLM, blocks are 16 tokens wide; here we use a token-by-token
    API to keep the example focused on the *management* contract, not the
    tensor math.
    """

    def __init__(self, num_blocks: int, block_size_tokens: int = 16) -> None:
        self.block_size = block_size_tokens
        self.pool: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_ids: list[int] = list(range(num_blocks))
        self.tables: dict[int, BlockTable] = {}
        print(f"[init] pool={num_blocks} blocks, block_size={block_size_tokens} tokens")

    def _alloc(self) -> int:
        if not self.free_ids:
            raise MemoryError("OOM: no free KV blocks")
        bid = self.free_ids.pop(0)
        self.pool[bid].ref_count = 1
        return bid

    def _free_chain(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            self.pool[bid].ref_count -= 1
            if self.pool[bid].ref_count <= 0:
                self.pool[bid].ref_count = 0
                self.pool[bid].is_full = False
                self.free_ids.append(bid)

    def begin_sequence(self, seq_id: int) -> None:
        self.tables[seq_id] = BlockTable(seq_id=seq_id)
        # Pre-allocate one block so the first tokens don't immediately OOM
        self.tables[seq_id].append_block(self._alloc())

    def append_token(self, seq_id: int) -> None:
        tbl = self.tables[seq_id]
        # Naive: alloc new block per token (in real vLLM, per block_size tokens)
        # We mimic "block full" by counting tokens
        last = tbl.block_ids[-1]
        # mark full every block_size tokens (toy)
        self.pool[last].is_full = True
        tbl.append_block(self._alloc())

    def fork(self, parent_seq: int, child_seq: int) -> None:
        """Copy-on-Write fork: child shares parent's blocks (ref_cnt++)."""
        parent = self.tables[parent_seq]
        child = BlockTable(seq_id=child_seq, block_ids=list(parent.block_ids))
        for bid in child.block_ids:
            self.pool[bid].ref_count += 1
        self.tables[child_seq] = child
        print(f"[fork] {parent_seq} -> {child_seq}, shared {len(child.block_ids)} blocks")

    def end_sequence(self, seq_id: int) -> None:
        self._free_chain(self.tables.pop(seq_id).block_ids)

    def stats(self) -> dict:
        free = len(self.free_ids)
        used = len(self.pool) - free
        return {
            "total_blocks": len(self.pool),
            "free": free,
            "used": used,
            "utilization": round(used / len(self.pool), 3),
        }


def main() -> None:
    bm = BlockManager(num_blocks=20, block_size_tokens=16)

    # Sequence A: short prompt
    bm.begin_sequence(seq_id=1)
    for _ in range(2):
        bm.append_token(1)
    print("[A] block table:", [b for b in bm.tables[1].block_ids])

    # Sequence B: fork from A (e.g. parallel sampling / beam) -> shares KV
    bm.fork(parent_seq=1, child_seq=2)
    for _ in range(3):
        bm.append_token(2)
    print("[B] block table:", [b for b in bm.tables[2].block_ids])

    print("stats after fork:", bm.stats())

    bm.end_sequence(1)
    print("stats after end(A):", bm.stats())
    bm.end_sequence(2)
    print("stats after end(B):", bm.stats())

    # Naive contiguous allocator would have needed len(A) + len(B) blocks.
    # Paged + CoW only needs max(len(A), len(B)) + tiny overhead.


if __name__ == "__main__":
    main()
