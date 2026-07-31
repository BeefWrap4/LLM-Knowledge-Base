# Ch17 — 大模型评估体系

> 教程: [`../tutorial/17_大模型评估体系.md`](../tutorial/17_大模型评估体系.md)

## 例子

| Tier | Files | 主题 |
|------|-------|------|
| llm | 14 | Langfuse Python SDK v4 / Phoenix Evals v3 / Ragas v0.4 / TruLens / DeepEval DAG / OpenAI Responses |

## 快速开始

```powershell
Set-Location code
$env:LLM_MOCK = "1"
python ch17_evaluation/llm/12_langfuse_v3.py
```

> `12_langfuse_v3.py` 为兼容既有教程链接保留文件名，内容已使用当前 Python SDK v4。
> 未设置 `LLM_MOCK` 时也默认离线；真实模式必须显式设置 `LLM_MOCK=0`。真实模式缺少凭据、
> 可选依赖或本地模型时会失败，不会静默退回模拟结果。

## 关联章节

- Ch20: 可观测性
- Ch14: RAG 评估
- Ch25: 在线 SLO 评估
