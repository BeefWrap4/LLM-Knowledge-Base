# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.3.2 ViT 代码实现 - PatchEmbedding + VisionTransformer
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 03_vit_implementation.py
# expected_runtime: 5-10s
# expected_output: ViT 输出的 logits 形状
# ---
# See: ../tutorial/21_多模态大模型.md#21-3-2-vit-代码实现
# Interview hooks:
#   1. ViT 如何将图像转换为序列？
#   2. [CLS] token 与全局平均池化在分类头设计上各有什么优劣？
#   3. ViT 处理不同分辨率图像的关键技巧（位置编码插值）是什么？

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """将图像切分为 patch 并线性投影"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 使用卷积实现 patch 切分（高效实现）
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.proj(x)  # [B, 768, 14, 14]
        x = x.flatten(2)  # [B, 768, 196]
        x = x.transpose(1, 2)  # [B, 196, 768]
        return x


class TransformerBlock(nn.Module):
    """单层 Transformer（Pre-LN 风格）"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-Attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """简化版 ViT-Base"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        # Transformer 编码器层
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # [B, 196, 768]
        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 197, 768]
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Transformer 编码
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # 取 [CLS] token 输出用于分类
        cls_output = x[:, 0]  # [B, 768]
        return self.head(cls_output)


def main():
    torch.manual_seed(0)
    # 构造一个 mini ViT 以保证快速运行
    vit = VisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=96,
        depth=2,
        num_heads=4,
        num_classes=10,
    )
    img = torch.randn(2, 3, 32, 32)
    logits = vit(img)
    print(f"Output logits shape: {tuple(logits.shape)}")
    # 验证参数数量
    n_params = sum(p.numel() for p in vit.parameters())
    print(f"ViT mini params: {n_params:,}")
    # 反向传播测试
    loss = logits.sum()
    loss.backward()
    print("Backward OK")


if __name__ == "__main__":
    main()
    print("OK")
