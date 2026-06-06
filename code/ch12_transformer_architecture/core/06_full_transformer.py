# ---
# chapter: 12
# topic: Transformer与大模型原理
# section: 12.9 完整 Transformer PyTorch 实现
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: torch>=2.0
# run: python 06_full_transformer.py
# expected_runtime: <10s (CPU)
# expected_output: 完整 Transformer 前向传播, 输出 shape (batch, tgt_len, tgt_vocab_size)
# ---
# See: ../tutorial/12_Transformer与大模型原理.md (Section 12.9)
# Interview hooks:
#   1. Encoder 与 Decoder 的核心结构差异是什么？为什么 Decoder 多一个 Cross-Attention？
#   2. Pre-LN 与 Post-LN 在训练稳定性上的权衡？
#   3. 为何嵌入要乘以 sqrt(d_model)？位置编码为什么可以与嵌入相加？

# ==== 自包含的依赖 (源自教程 12.2-12.4) ====

import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, h, n, d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        output = self.W_O(attn_output)
        output = self.dropout(output)
        return output, attn_weights


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""

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
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==== 12.9 节主代码 ====

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer — 包含 Self-Attention + FFN"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-LN 结构: Norm → Sublayer → Residual
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer — 包含 Masked Self-Attn + Cross Attn + FFN"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention
        self_attn_out, _ = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask
        )
        x = x + self_attn_out

        # Cross Attention (Q from decoder, K/V from encoder)
        cross_attn_out, _ = self.cross_attn(
            self.norm2(x), encoder_output, encoder_output, src_mask
        )
        x = x + cross_attn_out

        # FFN
        ffn_out = self.ffn(self.norm3(x))
        x = x + ffn_out

        return x


class Transformer(nn.Module):
    """完整 Transformer 模型"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, max_len=5000,
                 dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # 编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 解码器
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_layer(decoder_output)


def generate_causal_mask(seq_len, device):
    """生成 Decoder 自回归 causal mask (下三角)."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


if __name__ == "__main__":
    # 玩具 demo: src/tgt 词汇表 100, 序列长度 6
    torch.manual_seed(42)
    src_vocab, tgt_vocab = 100, 100
    batch, src_len, tgt_len = 2, 6, 5

    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_len=50,
        dropout=0.0,
    )

    src = torch.randint(0, src_vocab, (batch, src_len))
    tgt = torch.randint(0, tgt_vocab, (batch, tgt_len))
    tgt_mask = generate_causal_mask(tgt_len, src.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    out = model(src, tgt, tgt_mask=tgt_mask)
    print(f"src shape:  {src.shape}")
    print(f"tgt shape:  {tgt.shape}")
    print(f"output shape: {out.shape}  # 期望: torch.Size([{batch}, {tgt_len}, {tgt_vocab}])")
    assert out.shape == (batch, tgt_len, tgt_vocab)

    # 测试编码器独立
    enc_out = model.encode(src)
    print(f"encoder output shape: {enc_out.shape}  # 期望: ({batch}, {src_len}, 64)")
    assert enc_out.shape == (batch, src_len, 64)

    print("\nOK")
