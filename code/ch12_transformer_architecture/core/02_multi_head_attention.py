# ---
# chapter: 12
# topic: Multi-Head Attention (MHA)
# section: 12.3
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: torch
# run: python 02_multi_head_attention.py
# expected_runtime: <10s
# ---
#
# See: ../tutorial/Ch12_Transformer与大模型原理.md §12.3
# Cross-refs:
#   - §12.2.5 Scaled Dot-Product Attention (基础)
#   - Ch16.4 Flash Attention (生产级 MHA)
#
# Interview hooks:
#   - "为什么需要多头?"  →  不同 head 学习不同子空间, 增加表达力
#   - "head_dim 怎么定?" →  d_model / num_heads (LLaMA: 128)
#   - "GQA / MQA 是什么?" →  Grouped/Multi-Query Attention, 减少 KV 显存

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """教科书 MHA 实现."""

    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 4 个线性投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        # 1. 投影
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # shape: (batch, num_heads, seq_len, d_k)

        # 2. Scaled dot-product attention (per-head)
        # 用 PyTorch 2.0+ 内置 (更高效)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=mask, dropout_p=0.0
            )
        else:
            # 回退: 手写
            d_k = Q.size(-1)
            scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attn_w = torch.softmax(scores, dim=-1)
            attn_out = attn_w @ V

        # 3. 合并多头: (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # 4. 输出投影
        return self.W_o(attn_out)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, seq_len, d_model, num_heads = 2, 10, 512, 8
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)

    out = mha(x)
    print(f"input shape:  {x.shape}")
    print(f"output shape: {out.shape}  # 应与 input 一致")
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"

    # 参数数量
    n_params = sum(p.numel() for p in mha.parameters())
    print(f"参数量: {n_params:,}  # 4 * d_model^2 = 4 * 512^2 = 1,048,576")

    # 测试 causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    out_masked = mha(x, mask=causal_mask)
    print(f"causal 输出 shape: {out_masked.shape}")

    print("\nOK")
