# ch22_agent_tools 代码伴侣

> 本目录是运行分组，不是主题身份。示例的规范归属以 [`code/TOPIC_MANIFEST.json`](../TOPIC_MANIFEST.json) 中的稳定 `topic_id` 和 `chapter` 为准。

## 对应章节

- [第 22 章 Agent 基础与工具调用](../../22_Agent基础与工具调用.md)
- [第 23 章 MCP、A2A 与 Skills 协议生态](../../23_MCP_A2A与Skills协议生态.md)
- [第 24 章 Agent 工作流编排与多智能体](../../24_Agent工作流编排与多智能体.md)
- [第 25 章 可恢复 Agent 运行时](../../25_可恢复Agent运行时.md)
- [第 26 章 Agent 记忆与个性化](../../26_Agent记忆与个性化.md)
- [第 38 章 大模型与 Agent 安全](../../38_大模型与Agent安全.md)

## 示例统计

- 共 22 个示例：llm=22
- 每个示例保留 `# ---` metadata、`if __name__ == "__main__"` 入口和 `OK` 成功标记。

## 运行与验收

```powershell
python code/scripts/run_all_examples.py --chapter ch22 --tier core
python code/scripts/run_all_examples.py --chapter ch22 --tier llm
python code/scripts/run_all_examples.py --chapter ch22 --tier gpu
```

默认 LLM/GPU 验收使用 mock 或条件跳过；真实 API、GPU、模型下载和付费调用必须显式启用。
