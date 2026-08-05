# ch27_llm_frameworks 代码伴侣

> 本目录是运行分组，不是主题身份。示例的规范归属以 [`code/TOPIC_MANIFEST.json`](../TOPIC_MANIFEST.json) 中的稳定 `topic_id` 和 `chapter` 为准。

## 对应章节

- [第 24 章 Agent 工作流编排与多智能体](../../24_Agent工作流编排与多智能体.md)
- [第 26 章 Agent 记忆与个性化](../../26_Agent记忆与个性化.md)
- [第 27 章 LLM 框架与平台选型](../../27_LLM框架与平台选型.md)
- [第 30 章 SFT、LoRA 与 QLoRA](../../30_SFT_LoRA与QLoRA.md)

## 示例统计

- 共 37 个示例：llm=37
- 每个示例保留 `# ---` metadata、`if __name__ == "__main__"` 入口和 `OK` 成功标记。

## 运行与验收

```powershell
python code/scripts/run_all_examples.py --chapter ch27 --tier core
python code/scripts/run_all_examples.py --chapter ch27 --tier llm
python code/scripts/run_all_examples.py --chapter ch27 --tier gpu
```

默认 LLM/GPU 验收使用 mock 或条件跳过；真实 API、GPU、模型下载和付费调用必须显式启用。
