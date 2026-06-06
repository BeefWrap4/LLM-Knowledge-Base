# ---
# chapter: 12
# topic: Transformer与大模型原理
# section: 12.4.4 位置编码 (Positional Encoding)
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: torch>=2.0
# run: python 03_sinusoidal_positional_encoding.py
# expected_runtime: <5s (CPU)
# expected_output: PE 矩阵 shape (1, max_len, d_model) 且每对 (2i, 2i+1) 行为正弦/余弦
# ---
# See: ../tutorial/12_Transformer与大模型原理.md (Section 12.4.4)
# Interview hooks:
#   1. Self-Attention 本身是置换等变的，为什么需要位置编码？
#   2. sin/cos 位置编码与 RoPE 的根本差异是什么？
#   3. RoPE 为什么能天然支持相对位置？它如何被用于长上下文外推？

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """原版正弦位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    d_model = 16
    max_len = 50
    pe_layer = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)

    # 验证 PE 矩阵
    pe = pe_layer.pe.squeeze(0)  # (max_len, d_model)
    print(f"PE shape: {pe.shape}  # 期望: torch.Size([{max_len}, {d_model}])")
    assert pe.shape == (max_len, d_model)

    # 偶数维 = sin, 奇数维 = cos
    pos_0_sin = pe[0, 0].item()
    pos_0_cos = pe[0, 1].item()
    print(f"PE[0, 0] (sin 通道): {pos_0_sin:.4f}  # 期望 ≈ 0.0 (sin(0)=0)")
    print(f"PE[0, 1] (cos 通道): {pos_0_cos:.4f}  # 期望 ≈ 1.0 (cos(0)=1)")
    assert abs(pos_0_sin) < 1e-5
    assert abs(pos_0_cos - 1.0) < 1e-5

    # 验证高频与低频通道
    # 低频 (i=0): 周期 = 2π * 10000^(0/d_model) = 2π * 1 = 2π
    # 高频 (i=d_model-2): 周期 = 2π * 10000^((d_model-2)/d_model) ≈ 2π * 10000
    print(f"\n低频通道 (i=0)   周期对应数值: {pe[:5, 0].tolist()}")
    print(f"高频通道 (i=14)  周期对应数值: {pe[:5, 14].tolist()}")
    print("  # 低频变化慢(长周期), 高频变化快(短周期)")

    # 验证: 不同位置编码唯一
    assert not torch.allclose(pe[0], pe[1]), "位置 0 和 1 编码应不同"
    assert not torch.allclose(pe[5], pe[10]), "位置 5 和 10 编码应不同"

    # 验证: 嵌入 + 位置编码
    batch, seq_len = 2, 10
    x = torch.randn(batch, seq_len, d_model)
    out = pe_layer(x)
    print(f"\n输入 shape:   {x.shape}")
    print(f"输出 shape:   {out.shape}  # 应与输入一致")

    print("\nOK")
