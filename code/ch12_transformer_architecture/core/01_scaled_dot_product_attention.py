# ---
# chapter: 12
# topic: Scaled Dot-Product Attention (PyTorch 实现)
# section: 12.2.5
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: torch
# run: python 01_scaled_dot_product_attention.py
# expected_runtime: <5s
# ---
#
# See: ../tutorial/Ch12_Transformer与大模型原理.md §12.2.5
# Cross-refs:
#   - §12.2.1 Self-Attention 公式
#   - §12.3.2 Multi-Head Attention (consumes this output)
#   - Ch16.4 Flash Attention (生产级替代)
#
# Interview hooks:
#   - "为什么除以 sqrt(d_k)?"  →  数值稳定性, 防止 softmax 饱和
#   - "Q/K/V 从哪来?"          →  同一序列的三个不同线性投影
#   - "Mask 怎么用?"            →  Decoder 自回归: 上三角 mask 防止看到未来

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """教科书实现. 生产请用 F.scaled_dot_product_attention (PyTorch 2.0+).

    Args:
        Q: (batch, n_queries, d_k)
        K: (batch, n_keys, d_k)
        V: (batch, n_values, d_v)
        mask: (batch, n_queries, n_keys) or broadcastable, True=keep, False=mask

    Returns:
        output: (batch, n_queries, d_v)
        attn_weights: (batch, n_queries, n_keys)
    """
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return attn_weights @ V, attn_weights


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, n_q, n_k, d_k, d_v = 2, 4, 6, 8, 10

    Q = torch.randn(batch, n_q, d_k)
    K = torch.randn(batch, n_k, d_k)
    V = torch.randn(batch, n_k, d_v)

    out, attn = scaled_dot_product_attention(Q, K, V)
    print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    print(f"output shape: {out.shape}    # 期望: torch.Size([2, 4, 10])")
    print(f"attn shape:   {attn.shape}   # 期望: torch.Size([2, 4, 6])")

    # 验证: attention weights 每行和为 1
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"attention weights 行和应为 1, 实际 {row_sums}"
    print(f"attn row sums: {row_sums[0]}  # 全部接近 1.0")

    # 验证: output 形状正确
    assert out.shape == (batch, n_q, d_v)

    # 测试 causal mask (Decoder 自回归场景)
    print("\n--- Causal mask 测试 ---")
    causal_mask = torch.tril(torch.ones(n_q, n_k))  # 下三角 = 1
    out_masked, attn_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print(f"masked output shape: {out_masked.shape}")
    # 上三角位置的 attention 应为 0 (被 mask 掉)
    upper_tri = torch.triu(torch.ones(n_q, n_k), diagonal=1).bool()
    assert (attn_masked[:, upper_tri] < 1e-6).all(), "上三角 mask 失败"
    print(f"上三角位置 attention ~ 0: {attn_masked[0, :, 5].tolist()}")
    print("  # 第 0 行只能看 K[0], 第 4 行可看 K[0..4]")

    print("\nOK")
