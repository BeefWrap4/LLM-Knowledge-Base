# ---
# chapter: 11
# topic: 深度学习与PyTorch
# section: 11.4.4 Attention 机制的引入 (nn.LSTM)
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: torch
# run: python 07_lstm.py
# expected_runtime: <5s
# expected_output: LSTM output/hidden/cell shape
# ---
# See: ../tutorial/11_深度学习与PyTorch.md#11.4.4-attention-机制的引入
#
# Interview hooks:
#  1. LSTM 为何能缓解 RNN 的梯度消失? 细胞状态加性更新 vs 隐藏状态乘性累积?
#  2. 双向 LSTM (bidirectional=True) 的 hidden 维度为什么是 num_layers*2?
#  3. batch_first=True 时输入/输出/隐藏状态的 shape 顺序?
import torch
import torch.nn as nn

# PyTorch 内置 LSTM
lstm = nn.LSTM(
    input_size=128,  # 输入特征维度
    hidden_size=256,  # 隐藏状态维度
    num_layers=2,  # LSTM 层数
    batch_first=True,  # 输入格式 (batch, seq, feature)
    dropout=0.3,  # 层间 dropout
    bidirectional=True,  # 双向 LSTM
)


if __name__ == "__main__":
    # 输入: (batch_size, seq_len, input_size)
    x = torch.randn(32, 50, 128)
    output, (hidden, cell) = lstm(x)

    # output: (32, 50, 512) — 双向，hidden_size*2
    # hidden: (4, 32, 256) — num_layers*2, batch, hidden_size
    print(f"输出 shape: {output.shape}")
    print(f"hidden shape: {hidden.shape}")
    print(f"cell shape: {cell.shape}")
    print("OK")
