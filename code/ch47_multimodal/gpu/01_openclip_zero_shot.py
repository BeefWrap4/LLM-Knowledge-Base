# ---
# chapter: 47
# topic: 多模态表征与多模态大模型
# topic_id: multimodal.openclip_zero_shot
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch, open_clip, pillow
# run: CH21_OPENCLIP_RUN=1 CH21_IMAGE=/path/to/image.jpg python 01_openclip_zero_shot.py
# expected_runtime: depends on model cache, hardware, and input
# expected_output: 当前输入图像对三个候选文本的相似度概率
# ---
# See: ../../../47_多模态表征与多模态大模型.md
# Interview hooks:
#   1. CLIP 的双塔结构如何实现零样本图像分类？
#   2. 余弦相似度与可学习 logit scale 在对比学习中起什么作用？
#   3. 零样本标签模板为何需要在目标数据上验证？

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu, skip_if_mock


def main() -> None:
    """在显式确认后，用真实 OpenCLIP 权重和本地图像做零样本分类。"""
    if skip_if_mock("OpenCLIP weights, an NVIDIA GPU, and a local input image"):
        return
    if os.environ.get("CH21_OPENCLIP_RUN") != "1":
        print("[SKIP] Set CH21_OPENCLIP_RUN=1 after reviewing model download and input path.")
        return

    require_nvidia_gpu(min_vram_gb=4)
    image_path = Path(os.environ.get("CH21_IMAGE", "")).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError("CH21_IMAGE must point to an existing local image")

    try:
        import open_clip
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install open_clip_torch and Pillow for this real example") from exc

    model_name = os.environ.get("CH21_OPENCLIP_MODEL", "ViT-B-32")
    pretrained = os.environ.get("CH21_OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")
    labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.eval().to("cuda")
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to("cuda")
    text_tokens = tokenizer(labels).to("cuda")

    with torch.no_grad():
        image_features = F.normalize(model.encode_image(image), dim=-1)
        text_features = F.normalize(model.encode_text(text_tokens), dim=-1)
        probabilities = ((image_features @ text_features.T) * model.logit_scale.exp()).softmax(
            dim=-1
        )

    print(f"model={model_name}, pretrained={pretrained}, image={image_path.name}")
    for label, probability in zip(labels, probabilities[0], strict=True):
        print(f"  {label}: {probability.item():.4f}")


if __name__ == "__main__":
    main()
    print("OK")
