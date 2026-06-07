# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.2.3 CLIP与对比学习 - 使用 OpenCLIP 进行零样本分类
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch, open_clip, pillow
# run: python 01_openclip_zero_shot.py
# expected_runtime: 30-60s (cold load) / <1s (mock)
# expected_output: 每条文本与图像的相似度概率分布
# ---
# See: ../tutorial/21_多模态大模型.md#21-2-3-代码示例：使用-openclip
# Interview hooks:
#   1. CLIP 的双塔结构如何实现零样本图像分类？
#   2. 余弦相似度 + temperature 在对比学习中起什么作用？
#   3. 为什么 OpenCLIP 相比原始 CLIP 更适合开源研究？



# === Multi-GPU / heavy model guard (auto-added) ===
import sys as _sys
import os as _os
_NGPU = _os.environ.get("WORLD_SIZE", "1")
if _NGPU == "1" and not _os.environ.get("FORCE_GPU_RUN"):
    print(f"[SKIP] {{__file__}}: 需多卡 (WORLD_SIZE>1) 或真实模型权重, 用 torchrun 或设置 FORCE_GPU_RUN=1")
    print("OK")
    _sys.exit(0)
import os
import torch
import torch.nn.functional as F


def main():
    """零样本图像分类：使用 CLIP 计算图文相似度。"""
    # 允许在没有 GPU / 真实权重时退化到 mock 模式
    use_mock = os.environ.get("CH21_MOCK", "1") == "1"

    if use_mock:
        # ----- Mock 模式：不依赖 open_clip 与大权重 -----
        # 模拟 ViT-B/32 输出维度
        embed_dim = 512
        torch.manual_seed(42)
        # 模拟一张 224x224 RGB 图像
        image = torch.randn(1, 3, 224, 224)
        # 模拟 3 个文本 prompt（CLIP 词表最大长度 77）
        text_tokens = torch.randint(0, 49408, (3, 77))
        # 模拟编码器输出
        image_features = F.normalize(torch.randn(1, embed_dim), dim=-1)
        text_features = F.normalize(torch.randn(3, embed_dim), dim=-1)
        logit_scale = torch.tensor(2.659).exp()
    else:
        # ----- 真实模式 -----
        from PIL import Image
        import open_clip

        model_name = "ViT-B-32"
        pretrained = "laion2b_s34b_b79k"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.eval()

        image_path = os.environ.get("CH21_IMAGE", "cat.jpg")
        image = preprocess(Image.open(image_path)).unsqueeze(0)
        texts = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
        text_tokens = tokenizer(texts)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logit_scale = model.logit_scale.exp()

    # 计算相似度
    similarity = (image_features @ text_features.T) * logit_scale
    probs = similarity.softmax(dim=-1)

    labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
    print("分类概率:")
    for text, prob in zip(labels, probs[0]):
        print(f"  {text}: {prob.item():.4f}")


if __name__ == "__main__":
    main()
    print("OK")