# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.4.5 LoRA/QLoRA 参数配置详解
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (none - pure config)
# run: python 20_llama_factory_config_presets.py
# expected_runtime: <1s
# expected_output: config presets dict
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.4.5
# Interview hooks:
#   1. lora_rank 与 lora_alpha 的关系是什么？缩放因子如何影响训练？
#   2. gradient_accumulation_steps 与 per_device_train_batch_size 如何配合？
# ✅ 最佳实践：不同场景的参数组推荐
configs = {
    "快速实验": {
        "finetuning_type": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_target": "q_proj,v_proj",
        "per_device_train_batch_size": 4,
        "learning_rate": 2e-4,
        "num_train_epochs": 2,
    },
    "生产级微调": {
        "finetuning_type": "lora",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_target": "q_proj,k_proj,v_proj,o_proj",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
    },
    "低显存QLoRA": {
        "finetuning_type": "lora",
        "quantization_bit": 4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
    },
}

import json

print(json.dumps(configs, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    print("OK")
