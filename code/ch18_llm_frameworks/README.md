# Ch18 — LLM 工程框架实战

> 教程: [`../tutorial/Ch18_LLM工程框架实战.md`](../tutorial/Ch18_LLM工程框架实战.md)

| Tier | Files | 主题 |
|------|-------|------|
| llm | 37 | LangChain, LangGraph, LlamaIndex, Pydantic AI, Strands, OpenAI Agents 等 |

## 离线验收

```bash
make install-llm
LLM_MOCK=1 python scripts/run_all_examples.py --tier llm --chapter ch18
```

离线验收不会调用模型提供商或创建远端资源；依赖未安装的可选示例应输出 `[SKIP]`。

## 真实调用

真实调用必须显式设置 `LLM_MOCK=0`，并配置对应提供商的 key。OpenAI 示例默认使用
`OPENAI_MODEL=gpt-5.6`，可按当前模型目录与评测结果覆盖；Strands/Bedrock 示例必须从
`BEDROCK_MODEL_ID` 读取部署区域中已获授权的模型 ID。
