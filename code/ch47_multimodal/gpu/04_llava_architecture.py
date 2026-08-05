# ---
# chapter: 47
# topic: 多模态表征与多模态大模型
# topic_id: multimodal.llava_architecture
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 04_llava_architecture.py
# expected_runtime: <5s
# expected_output: 教学张量的投影与拼接形状
# ---
# See: ../../../47_多模态表征与多模态大模型.md
# Interview hooks:
#   1. 视觉投影层为何需要把 vision hidden size 映射到 LLM hidden size？
#   2. 图像 patch token 可以怎样插入文本 embedding 序列？
#   3. 完整 LLaVA 还需要哪些 processor、mask、label 与 checkpoint 约束？

import torch
import torch.nn as nn


def main() -> None:
    """只演示投影与序列拼接，不加载权重，也不声称是可生成的 LLaVA。"""
    torch.manual_seed(42)
    batch_size, patch_count = 1, 16
    vision_hidden_size, llm_hidden_size = 32, 64
    text_length = 8

    vision_features = torch.randn(batch_size, patch_count, vision_hidden_size)
    projector = nn.Sequential(
        nn.Linear(vision_hidden_size, llm_hidden_size),
        nn.GELU(),
        nn.Linear(llm_hidden_size, llm_hidden_size),
    )
    projected = projector(vision_features)

    text_embeddings = torch.randn(batch_size, text_length, llm_hidden_size)
    image_position = 1
    inputs_embeds = torch.cat(
        [
            text_embeddings[:, :image_position],
            projected,
            text_embeddings[:, image_position:],
        ],
        dim=1,
    )

    assert projected.shape == (batch_size, patch_count, llm_hidden_size)
    assert inputs_embeds.shape == (
        batch_size,
        text_length + patch_count,
        llm_hidden_size,
    )
    print("[STRUCTURE ONLY] No tokenizer, checkpoint, attention mask, labels, or generation.")
    print(f"projected={tuple(projected.shape)}, inputs_embeds={tuple(inputs_embeds.shape)}")


if __name__ == "__main__":
    main()
    print("OK")
