# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.2.4 简化版 CLIP 训练 - 双塔 + InfoNCE Loss
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 02_simple_clip.py
# expected_runtime: <5s
# expected_output: InfoNCE 损失值
# ---
# See: ../tutorial/21_多模态大模型.md#21-2-4-实现-clip-训练（简化版）
# Interview hooks:
#   1. InfoNCE Loss 的对称形式 (I->T + T->I) 有什么意义？
#   2. 为什么温度参数 τ 需要可学习？
#   3. 对比学习为什么需要大 batch size？

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCLIP(nn.Module):
    """简化版 CLIP 双塔模型"""

    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # 投影层：将编码器输出映射到统一的嵌入维度
        self.image_proj = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim)
        # 可学习的温度参数（初始化为 log(1/0.07) ≈ 2.659）
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

    def encode_image(self, image):
        features = self.image_encoder(image)
        features = self.image_proj(features)
        return F.normalize(features, dim=-1)

    def encode_text(self, text):
        features = self.text_encoder(text)
        features = self.text_proj(features)
        return F.normalize(features, dim=-1)

    def forward(self, image, text):
        # 编码
        image_embeds = self.encode_image(image)  # [B, d]
        text_embeds = self.encode_text(text)  # [B, d]
        # 计算相似度矩阵
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T  # [B, B]
        logits_per_text = logits_per_image.T
        # 标签：对角线为正样本（第i张图对应第i条文本）
        labels = torch.arange(len(image), device=image.device)
        # 双向 InfoNCE Loss
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2
        return loss, image_embeds, text_embeds


class DummyVisionEncoder(nn.Module):
    """占位视觉编码器，模拟 ViT 输出。"""

    def __init__(self, output_dim=768):
        super().__init__()
        self.output_dim = output_dim
        self.body = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, output_dim))

    def forward(self, x):
        # x: [B, 3, H, W] -> 平均池化到 [B, 3] -> Linear -> [B, output_dim]
        if x.dim() == 4:
            x = x.mean(dim=(2, 3))  # [B, 3]
        return self.body[:, 0:0](x) if False else self.body._modules["2"](x)


class DummyTextEncoder(nn.Module):
    """占位文本编码器，模拟 Transformer 输出。"""

    def __init__(self, vocab_size=1000, output_dim=512, max_len=77):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Embedding(vocab_size, output_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, ids):
        # ids: [B, L]
        x = self.embed(ids)  # [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.pool(x).squeeze(-1)  # [B, D]
        return x


def main():
    torch.manual_seed(0)
    # 用 dummy encoder 构造 SimpleCLIP，做一次 forward
    vision_encoder = nn.Linear(3, 768)  # 简化: 3 -> 768
    text_encoder = nn.Embedding(1000, 512)

    class VE(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.output_dim = 768

        def forward(self, x):
            if x.dim() == 4:
                x = x.mean(dim=(2, 3))
            return self.inner(x)

    class TE(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.output_dim = 512

        def forward(self, ids):
            return self.inner(ids).mean(dim=1)

    model = SimpleCLIP(VE(vision_encoder), TE(text_encoder), embed_dim=256)
    B = 4
    images = torch.randn(B, 3, 32, 32)
    texts = torch.randint(0, 1000, (B, 16))
    loss, img_emb, txt_emb = model(images, texts)
    print(f"InfoNCE Loss: {loss.item():.4f}")
    print(f"Image embeds shape: {tuple(img_emb.shape)}")
    print(f"Text embeds shape: {tuple(txt_emb.shape)}")
    # 反向传播验证
    loss.backward()
    print("Backward OK")


if __name__ == "__main__":
    main()
    print("OK")
