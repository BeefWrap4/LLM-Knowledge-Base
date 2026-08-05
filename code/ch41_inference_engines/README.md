# ch41_inference_engines 代码伴侣

> 本目录是运行分组，不是主题身份。示例的规范归属以 [`code/TOPIC_MANIFEST.json`](../TOPIC_MANIFEST.json) 中的稳定 `topic_id` 和 `chapter` 为准。

## 对应章节

- [第 40 章 推理内存、量化与批处理](../../40_推理内存量化与批处理.md)
- [第 41 章 高性能推理引擎与服务](../../41_高性能推理引擎与服务.md)
- [第 42 章 PD 分离推理与 KV 池化](../../42_PD分离推理与KV池化.md)

## 示例统计

- 共 12 个示例：gpu=12
- 每个示例保留 `# ---` metadata、`if __name__ == "__main__"` 入口和 `OK` 成功标记。

## 运行与验收

```powershell
python code/scripts/run_all_examples.py --chapter ch41 --tier core
python code/scripts/run_all_examples.py --chapter ch41 --tier llm
python code/scripts/run_all_examples.py --chapter ch41 --tier gpu
```

默认 LLM/GPU 验收使用 mock 或条件跳过；真实 API、GPU、模型下载和付费调用必须显式启用。
