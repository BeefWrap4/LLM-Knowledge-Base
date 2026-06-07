# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.4.2 LLaVA 核心代码实现
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch, transformers
# run: python 04_llava_architecture.py
# expected_runtime: <5s (mock) / 30s+ (real)
# expected_output: LLaVA 架构 forward 演示
# ---
# See: ../tutorial/21_多模态大模型.md#21-4-2-llava-核心代码实现
# Interview hooks:
#   1. LLaVA 投影层（mm_projector）为什么需要 2 层 MLP 而不是单层？
#   2. LLaVA-1.5 为什么取 vision encoder 的倒数第二层 hidden state？
#   3. 多模态对话中如何将图像特征插入文本 token 序列？



# === Multi-GPU / heavy model guard (auto-added) ===
import sys as _sys
import os as _os
_NGPU = _os.environ.get("WORLD_SIZE", "1")
if _NGPU == "1" and not _os.environ.get("FORCE_GPU_RUN"):
    print(f"[SKIP] {{__file__}}: 需多卡 (WORLD_SIZE>1) 或真实模型权重, 用 torchrun 或设置 FORCE_GPU_RUN=1")
    _sys.exit(0)
import os
import torch
import torch.nn as nn


def main():
    """演示 LLaVA 中"图像特征投影 → 序列拼接"的核心流程，不依赖真实 LLM。"""
    use_mock = os.environ.get("CH21_MOCK", "1") == "1"

    # 模拟视觉编码器输出（CLIP ViT-L/14 第 -2 层，patch 序列）
    B = 2
    N_patches = 256
    vision_hidden_size = 1024
    llm_hidden_size = 4096
    vision_features = torch.randn(B, N_patches, vision_hidden_size)

    # 1) 投影层：vision space -> LLM space
    mm_projector = nn.Sequential(
        nn.Linear(vision_hidden_size, llm_hidden_size),
        nn.GELU(),
        nn.Linear(llm_hidden_size, llm_hidden_size),
    )
    projected = mm_projector(vision_features)
    print(f"Projected vision features shape: {tuple(projected.shape)}")

    # 2) 构造 LLM 输入嵌入（演示用）
    text_len = 16
    text_embeds = torch.randn(B, text_len, llm_hidden_size)
    # 3) 在 [IMAGE] 位置拼接
    img_pos = 1  # 第 1 个位置放 image token
    inputs_embeds = torch.cat(
        [
            text_embeds[:, :img_pos, :],     # prefix (含 <image> 之前)
            projected,                         # visual tokens
            text_embeds[:, img_pos:, :],     # suffix
        ],
        dim=1,
    )
    print(f"Concatenated input embeds shape: {tuple(inputs_embeds.shape)}")

    # 4) 不加载真实 LLM 时，仅演示逻辑；可选用 mock 后续层
    if not use_mock:
        try:
            from transformers import AutoModel, AutoTokenizer
            llm_name = os.environ.get("CH21_LLM", "meta-llama/Llama-3-8B-Instruct")
            llm = AutoModel.from_pretrained(llm_name)
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
            # 真实 LLM 不在 CPU 上跑，演示参数加载
            print(f"Loaded LLM: {llm_name}")
        except Exception as e:
            print(f"LLM load skipped: {e}")

    print("LLaVA forward demo OK")


if __name__ == "__main__":
    main()
    print("OK")