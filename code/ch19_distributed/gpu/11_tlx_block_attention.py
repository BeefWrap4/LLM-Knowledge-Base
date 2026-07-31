# ---
# chapter: 19
# topic: TLX Block Attention 的固定块对角结构
# section: 19.9.3
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: 无（真实 kernel 需 NVIDIA Blackwell、TLX/triton-ext 与官方实现）
# run: python 11_tlx_block_attention.py
# expected_runtime: <1s
# expected_output: fixed block-diagonal attention pairs + explicit capability boundary
# ---
# See: ../tutorial/19_分布式训练系统.md#1993-tlx-block-attention固定块对角blackwell-专用
# Official source:
# https://pytorch.org/blog/tlx-block-attention-a-warp-specialized-blackwell-kernel-for-fixed-block-sparse-self-attention/
# Interview hooks:
# 1. TLX 的全名是什么，它与普通 Triton/FlexAttention 是什么关系？
# 2. 固定 64-token block-diagonal 约束为何能消除在线 softmax 修正和 LSE 存取？
# 3. 为什么官方 kernel 不能概括为任意 128K/1M 长上下文的通用加速器？
"""TLX Block Attention 的结构演示。

官方 TLX kernel 是面向 NVIDIA Blackwell ``sm_100+`` 的独立 Triton
Language Extensions 实现，适配固定 64-token block-diagonal attention
与 head dimension 64/128。它不是 ``torch.nn.attention.flex_attention`` 的
自动后端；本文件只验证稀疏结构，不冒充真实 TLX kernel 或性能基准。
"""


def fixed_block_attention_pairs(sequence_length: int, block_size: int = 64) -> list[tuple[int, int]]:
    """返回 token 级固定块对角注意力允许的 ``(query, key)`` 对。"""

    if sequence_length <= 0:
        raise ValueError("sequence_length 必须为正整数")
    if block_size <= 0:
        raise ValueError("block_size 必须为正整数")

    pairs: list[tuple[int, int]] = []
    for query_index in range(sequence_length):
        block_start = (query_index // block_size) * block_size
        block_end = min(block_start + block_size, sequence_length)
        pairs.extend((query_index, key_index) for key_index in range(block_start, block_end))
    return pairs


def main() -> None:
    sequence_length = 192
    block_size = 64
    pairs = fixed_block_attention_pairs(sequence_length, block_size)

    dense_pairs = sequence_length**2
    print("=== TLX Block Attention 结构边界 ===")
    print(f"sequence={sequence_length}, fixed_block={block_size}")
    print(f"block-diagonal pairs={len(pairs)}, dense pairs={dense_pairs}")
    print(f"结构稀疏率={1 - len(pairs) / dense_pairs:.1%}")
    print("真实运行前提: Blackwell sm_100+、TLX/triton-ext、官方 kernel 与受支持 shape")
    print("注意: 这是结构 smoke，不是 FlexAttention 调用，也不是性能实测")
    print("OK")


if __name__ == "__main__":
    main()
