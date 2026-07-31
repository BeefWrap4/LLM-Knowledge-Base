# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.6.4 多模态 SFT - 数据整理与配置教学骨架
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: Python standard library
# run: python 07_sft_training.py
# expected_runtime: <1s
# expected_output: 对话 batch 结构与教学配置
# ---
# See: ../tutorial/21_多模态大模型.md#21-6-4-多模态微调实战脚本
# Interview hooks:
#   1. 多模态 collator 要怎样共同处理图像、对话模板、mask 与 labels？
#   2. 梯度累积、checkpointing 与 ZeRO 的收益为何必须按模型和硬件实测？
#   3. 怎样证明一次 SFT 真正执行了 optimizer step 并产出可加载 checkpoint？

from typing import Any


def collate_structure(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """整理教学结构；真实 processor 还需生成 pixel_values、input_ids 与 labels。"""
    images = [item["image"] for item in batch]
    conversations = [
        [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["question"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]},
        ]
        for item in batch
    ]
    return {"images": images, "conversations": conversations}


def training_config_template() -> dict[str, Any]:
    """返回待按 checkpoint、数据量和硬件验证的配置骨架。"""
    return {
        "epochs": 1,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "distributed_strategy": "configure explicitly; not executed by this example",
        "output_dir": "./llava-lora-checkpoints",
    }


def main() -> None:
    batch = [
        {"image": "img_0.png", "question": "图中有几只猫？", "answer": "两只。"},
        {"image": "img_1.png", "question": "描述这张图。", "answer": "夕阳下的海面。"},
    ]
    collated = collate_structure(batch)
    config = training_config_template()
    assert len(collated["images"]) == len(collated["conversations"]) == 2
    assert collated["conversations"][0][0]["content"][0]["type"] == "image"
    print("[STRUCTURE ONLY] No processor, checkpoint, optimizer step, save, or reload occurred.")
    print(f"batch_size={len(collated['images'])}, config={config}")


if __name__ == "__main__":
    main()
    print("OK")
