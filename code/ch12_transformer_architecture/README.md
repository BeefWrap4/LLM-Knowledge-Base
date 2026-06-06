# Ch12 — Transformer 与大模型原理

> 教程: [`../tutorial/Ch12_Transformer与大模型原理.md`](../tutorial/Ch12_Transformer与大模型原理.md)

| Tier | Files | 主题 |
|------|-------|------|
| core | 4 | Attention, MHA, Positional Encoding, KV Cache |

## 快速开始

```bash
python ch12_transformer_architecture/core/01_scaled_dot_product_attention.py
```

## 核心公式

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

除以 sqrt(d_k) 是为了防止点积方差过大导致 softmax 梯度消失。
