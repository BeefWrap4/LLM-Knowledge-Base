# ch47_multimodal 代码伴侣

> 本目录是运行分组，不是主题身份。示例的规范归属以 [`code/TOPIC_MANIFEST.json`](../TOPIC_MANIFEST.json) 中的稳定 `topic_id` 和 `chapter` 为准。

## 对应章节

- [第 21 章 生产级 RAG 系统](../../21_生产级RAG系统.md)
- [第 47 章 多模态表征与多模态大模型](../../47_多模态表征与多模态大模型.md)
- [第 48 章 扩散模型与生成式视觉](../../48_扩散模型与生成式视觉.md)
- [第 49 章 世界模型、VLA 与具身智能](../../49_世界模型VLA与具身智能.md)

## 示例统计

- 共 11 个示例：gpu=11
- 每个示例保留 `# ---` metadata、`if __name__ == "__main__"` 入口和 `OK` 成功标记。

## 运行与验收

```powershell
python code/scripts/run_all_examples.py --chapter ch47 --tier core
python code/scripts/run_all_examples.py --chapter ch47 --tier llm
python code/scripts/run_all_examples.py --chapter ch47 --tier gpu
```

默认 LLM/GPU 验收使用 mock 或条件跳过；真实 API、GPU、模型下载和付费调用必须显式启用。
