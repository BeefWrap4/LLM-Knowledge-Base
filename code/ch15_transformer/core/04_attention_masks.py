# ---
# chapter: 15
# topic: Transformer 架构与实现
# topic_id: transformer.attention_masks
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: torch>=2.0
# run: python 04_attention_masks.py
# expected_runtime: <5s (CPU)
# expected_output: 因果掩码下三角矩阵 + 填充掩码正确广播
# ---
# See: ../../../15_Transformer架构与实现.md
# Interview hooks:
#   1. Padding Mask 和 Causal Mask 有什么区别？分别在什么场景使用？
#   2. Causal Mask 为什么能让 Decoder 保持自回归特性？
#   3. 如何把两种 mask 合并成单个 mask 用于一次性 attention 计算？

import torch


def create_causal_mask(seq_len):
    """创建下三角掩码 — Decoder 自注意力使用"""
    # mask[i,j] = True 表示位置 i 可以关注位置 j
    mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角矩阵
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq, pad_idx=0):
    """创建填充掩码 — 忽略 pad token"""
    # (batch, 1, 1, seq_len)，广播到 (batch, 1, seq_len, seq_len)
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # (batch, 1, 1, seq_len)


if __name__ == "__main__":
    # 因果掩码可视化
    causal = create_causal_mask(5)
    print("Causal Mask (下三角):")
    print(causal.int())
    # 输出:
    # tensor([[1, 0, 0, 0, 0],
    #         [1, 1, 0, 0, 0],
    #         [1, 1, 1, 0, 0],
    #         [1, 1, 1, 1, 0],
    #         [1, 1, 1, 1, 1]])

    # 验证: 上三角位置为 0
    for i in range(5):
        for j in range(5):
            if j > i:
                assert causal[i, j].item() == 0, f"causal[{i},{j}] 应为 0"
    print("\n  # 上三角=0 (被 mask), 下三角=1 (可关注)")

    # 填充掩码示例
    # 假设 pad_idx=0, 有效 token 标号非 0
    batch = 2
    seq = torch.tensor(
        [
            [1, 2, 3, 4, 0, 0],  # 后两个是 pad
            [5, 6, 0, 0, 0, 0],  # 后四个是 pad
        ]
    )
    pad_mask = create_padding_mask(seq, pad_idx=0)
    print(f"\nPadding mask shape: {pad_mask.shape}  # (batch, 1, 1, seq_len)")
    print(f"Padding mask[0, 0, 0, :]: {pad_mask[0, 0, 0, :].int().tolist()}")
    print("  # 1=有效 token, 0=pad (应忽略)")

    # 验证: 广播维度允许与 attention scores 相乘
    # attention scores: (batch, num_heads, seq_len, seq_len)
    scores = torch.randn(batch, 4, 6, 6)
    masked = scores.masked_fill(pad_mask == 0, float("-inf"))
    print(f"\nscores shape: {scores.shape}")
    print(f"masked scores[0, 0, 0, :]: {masked[0, 0, 0, :].tolist()}")
    print("  # pad 位置变为 -inf, softmax 后贡献为 0")

    # 验证: causal + padding 联合掩码
    seq_len = 6
    causal_2d = create_causal_mask(seq_len)  # (seq_len, seq_len)
    # 把 2D causal 扩展到 (1, 1, seq_len, seq_len) 并与 padding mask 相乘
    causal_4d = causal_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    combined = causal_4d.bool() & pad_mask.bool()  # 同时满足 causal 和 padding
    print(f"\n联合 mask shape: {combined.shape}")
    print(f"联合 mask[0, 0, 5, :]: {combined[0, 0, 5, :].int().tolist()}")
    print("  # 位置 5 是 pad → 全部 0; 非 pad 位置受 causal 约束")

    print("\nOK")
