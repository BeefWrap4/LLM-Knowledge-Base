# ch17_prompt_engineering 代码伴侣

> 本目录是运行分组，不是主题身份。示例的规范归属以 [`code/TOPIC_MANIFEST.json`](../TOPIC_MANIFEST.json) 中的稳定 `topic_id` 和 `chapter` 为准。

## 对应章节

- [第 17 章 Prompt Engineering](../../17_Prompt_Engineering.md)
- [第 18 章 Context Engineering](../../18_Context_Engineering.md)
- [第 28 章 Computer Use 与 GUI Agent](../../28_ComputerUse与GUIAgent.md)
- [第 32 章 推理模型与 Test-Time Compute](../../32_推理模型与Test_Time_Compute.md)
- [第 38 章 大模型与 Agent 安全](../../38_大模型与Agent安全.md)

## 示例统计

- 共 22 个示例：llm=22
- 每个示例保留 `# ---` metadata、`if __name__ == "__main__"` 入口和 `OK` 成功标记。

## 运行与验收

```powershell
python code/scripts/run_all_examples.py --chapter ch17 --tier core
python code/scripts/run_all_examples.py --chapter ch17 --tier llm
python code/scripts/run_all_examples.py --chapter ch17 --tier gpu
```

默认 LLM/GPU 验收使用 mock 或条件跳过；真实 API、GPU、模型下载和付费调用必须显式启用。
