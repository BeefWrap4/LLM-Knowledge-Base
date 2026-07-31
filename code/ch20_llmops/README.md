# Ch20 — LLMOps 与可观测性

> 教程: [`../../20_LLMOps与模型可观测性.md`](../../20_LLMOps与模型可观测性.md)

## 例子

| Tier | Files | 主题 |
|------|-------|------|
| llm | 25 | OTel GenAI SemConv / OpenInference / Prompt 与 Eval / 可注入 Rate Card / 发布回滚 |

## 快速开始

```bash
cd code/
python ch20_llmops/llm/19_otel_genai_telemetry.py
```

示例默认离线且不读取 API Key。真实模型或外部追踪平台仅在授权环境中同时显式设置
`LLM_MOCK=0` 与 `LLM_REAL_API=1`；生产 OTLP 导出另需 `OTEL_EXPORT_ENABLED=1`。模型 ID、价格、
上下文窗口和 SLO 阈值均应从当前供应商文档或业务配置注入。

## 关联章节

- Ch17: 评估是 Ops 一部分
- Ch18: 框架集成了 trace
- Ch25: 推理 SLO 接入
