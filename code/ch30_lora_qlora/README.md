# ch30_lora_qlora 代码伴侣

> 本目录是运行分组，不是主题身份。示例的规范归属以 [`code/TOPIC_MANIFEST.json`](../TOPIC_MANIFEST.json) 中的稳定 `topic_id` 和 `chapter` 为准。

## 对应章节

- [第 30 章 SFT、LoRA 与 QLoRA](../../30_SFT_LoRA与QLoRA.md)
- [第 31 章 偏好对齐与强化学习](../../31_偏好对齐与强化学习.md)
- [第 40 章 推理内存、量化与批处理](../../40_推理内存量化与批处理.md)
- [第 41 章 高性能推理引擎与服务](../../41_高性能推理引擎与服务.md)
- [第 46 章 端侧、浏览器与边缘 LLM](../../46_端侧浏览器与边缘LLM.md)

## 示例统计

- 共 15 个示例：gpu=15
- 每个示例保留 `# ---` metadata、`if __name__ == "__main__"` 入口和 `OK` 成功标记。

## 运行与验收

```powershell
python code/scripts/run_all_examples.py --chapter ch30 --tier core
python code/scripts/run_all_examples.py --chapter ch30 --tier llm
python code/scripts/run_all_examples.py --chapter ch30 --tier gpu
```

默认 LLM/GPU 验收使用 mock 或条件跳过；真实 API、GPU、模型下载和付费调用必须显式启用。
