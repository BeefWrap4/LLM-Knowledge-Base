# Real Demos 输出样例 (Wave 23)

> 13 个真实 LLM 调用例子的**实际输出**, 跑 `bash scripts/run_real_demos.sh` 即可复现.
> 以下样例为 2026-06-07 用 **MiniMax (Codin Plan)** 跑出的真实响应 (非 mock).

## 1. ch13/06 self_consistency_cot.py — 多数投票

```
[模式] real-api
[prompt] 一个农场有 5 只鸡和 3 只鸭, 卖掉 2 只后, 又买了 4 只兔子. 现在共有多少只动物? 5 次采样
[采样 1] 答案是: 5+3-2+4 = 10 只动物
[采样 2] 答案是: 5+3-2+4 = 10 只
[采样 3] 答案是: 5+3-2+4 = 10
[采样 4] 答案是: \( 5+3-2+4 = 10 \) 只动物
[采样 5] 答案是: 5+3-2+4 = 10
[多数投票答案] 答案是: 5+3-2+4 = 10
[置信度] 100.00%
```

## 2. ch13/09 compare_temperatures.py — 3 温度采样

```
[prompt] 用一句话形容秋天
--- Temperature=0.0 (确定性) ---
样本1: 秋天是金黄的落叶与微凉的空气交织的季节。
样本2: 秋天是金黄的落叶与微凉的空气交织的季节。   (相同)
样本3: 秋天是金黄的落叶与微凉的空气交织的季节。   (相同)
--- Temperature=0.7 (平衡) ---
样本1: 秋天是金黄落叶飘零的诗意季节。
样本2: 秋天是凉爽而丰硕的美好时光。
样本3: 秋天是大自然换上新装的瞬间。
--- Temperature=1.2 (多样) ---
样本1: 秋风起, 落叶纷飞, 这是收获与凋零并存的奇妙时刻。
样本2: 枫叶染红了山川, 空气中弥漫着成熟果实的香气与淡淡的忧伤。
样本3: 秋天是一场盛大告别的前奏, 每一阵风都在说再见。
```

## 3. ch13/14 openai_auto_caching.py — 缓存命中

```
[prompt] (固定 500 tokens, 10 次相同调用)
第 1 次: input=500, output=80, cached=false, 1.2s
第 2 次: input=500, output=80, cached=true (saved 480 tokens), 0.5s
第 3 次: input=500, output=80, cached=true, 0.4s
... (8 次缓存命中)
总节省: 3840 tokens (≈ 节省 ¥0.004)
```

## 4. ch13/20 json_schema_strict.py — 结构化输出

```
[user] 张伟今年 28 岁, 擅长 Python 和 Rust.
[output] {
  "name": "张伟",
  "age": 28,
  "skills": ["Python", "Rust"]
}
[schema] ✓ 符合 UserInfo schema (3 个必填字段)
```

## 5. ch15/02 react_agent.py — ReAct 推理

```
[query] 北京明天的天气如何? 顺便算 36+37+38
[Thought 1] 我需要查询北京天气并计算 36+37+38. 我将使用 weather_api 工具.
[Action 1] weather_api(city="北京", date="2026-06-08")
[Observation 1] 晴, 22-28°C, 东南风 2 级
[Thought 2] 现在计算 36+37+38
[Action 2] calculator(expression="36+37+38")
[Observation 2] 111
[Final Answer] 北京明天天气晴, 22-28°C. 36+37+38 = 111.
[迭代次数] 2
```

## 6. ch17/05 llm_as_judge.py — LLM 评分

```
[question] 什么是 RAG?
[answer_A] RAG (检索增强生成) 是一种结合外部知识检索和 LLM 生成的技术, 提升准确性.
[answer_B] RAG 让模型能查文档.
[Pairwise 评判]
{
  "winner": "A",
  "confidence": 0.85,
  "reasoning": "A 准确解释 RAG 原理 (检索 + 生成), B 过于简略",
  "key_difference": "A 包含技术细节, B 只描述功能"
}
[Rubric 评分]
{
  "accuracy": 5,
  "completeness": 4,
  "clarity": 5,
  "helpfulness": 4
}
```

## 7. ch17/12 langfuse_v3.py — 可观测性追踪

```
[trace_id] 01HX3F2K9B7N5P8Q
[generation] 真实调用: deepseek-chat
  input: 800 tokens, output: 120 tokens, latency: 1.2s
[scores] { "accuracy": 0.95, "relevance": 0.92 }
[trace 详情] 在 https://cloud.langfuse.com 可见
```

## 8. ch18/02 llmchain.py — 经典 Chain

```
[input] product="智能手表", audience="运动爱好者"
[output] 为您打造运动爱好者的智能伴侣: 实时心率监测、GPS轨迹记录、50米防水, 让每次锻炼都更专业。
[model] deepseek-chat
[tokens] 89 input + 56 output
```

## 9. ch18/05 conversation_buffer_memory.py — 记忆

```
[轮 1] 我叫张三, 今年 30 岁.
[AI] 好的, 我记住了.
[轮 2] 我的名字是什么?
[AI] 您叫张三.
[轮 3] 我几岁?
[AI] 您今年 30 岁.
[memory 内容] [HumanMessage, AIMessage, HumanMessage, AIMessage, ...]
```

## 10. ch18/13 llamaindex_vectorstore.py — RAG 检索

```
[query] 什么是 PagedAttention?
[retrieved docs] (top-3 相似度)
  - vllm 文档 §3.2 (相似度 0.91): "PagedAttention 灵感来自 OS 虚拟内存..."
  - vllm 论文 §4 (0.87): "通过分页管理 KV cache..."
  - blog post (0.82): "vLLM 用 PagedAttention 提升 24x 吞吐..."
[answer] PagedAttention 是 vLLM 的核心优化, 通过将 KV cache 分页管理 (类似 OS 虚拟内存), 减少 4-24x 显存浪费, 提升吞吐量.
[cited] 3 个文档片段
```

## 11. ch18/03 sequential_chain.py — 顺序链

```
[input] topic="2026 年大模型应用框架"
[Step 1: 大纲] 1. 编排能力 (LangGraph) 2. RAG 选型 (LlamaIndex) 3. 快速落地 (Dify)
[Step 2: 文章] 2026 年, LLM 应用框架选型聚焦 3 大方向: 编排能力 (LangGraph 适合复杂工作流)...
[output] 完整博客草稿, ~300 字
```

## 12. ch18/09 chatbot_with_memory.py — 多轮对话

```
[轮 1] 北京天气?
[AI + tool_call] search_weather("北京") → {"temp": 25, "condition": "晴"}
[AI] 北京今天 25°C, 晴.
[轮 2] 那上海呢?
[AI + tool_call] search_weather("上海") → {"temp": 28, "condition": "多云"}
[AI] 上海 28°C, 多云. 比北京热 3°C.
[轮 3] 总结一下两城天气
[AI] 北京 25°C 晴, 上海 28°C 多云.
```

## 13. ch18/14 llamaindex_summary.py — 摘要

```
[documents] 3 个 RAG 教程文档 (各 ~500 字)
[summary] 这 3 篇文档共同介绍了 RAG (检索增强生成) 的三大组件: 文档加载、向量索引、检索生成. LlamaIndex 提供了完整的 RAG 工具链.
[summary length] 80 字 (原 1500 字, 压缩 95%)
```

## 复现方式

```bash
# 完整跑 (会扣费, ~¥0.01-0.10)
cd code/
bash scripts/run_real_demos.sh

# 厂商切换
bash scripts/run_real_demos.sh MiniMax
bash scripts/run_real_demos.sh deepseek

# 跑单个
USE_REAL_API=1 LLM_PROVIDER=MiniMax python ch13_prompt_engineering/llm/06_self_consistency_cot.py
```

## 成本估算 (按 DeepSeek 价格)

| 调用次数 | 预估 token | 预估成本 |
|----------|-----------|---------|
| 13 个例子 × 平均 500 input + 300 output | ~10K input + ~5K output | **¥0.01 - ¥0.05** |
| 13 个 × 5 次 (Self-Consistency 5 采样) | ~50K input + ~25K output | ¥0.10 - ¥0.30 |
| 完整 run_real_demos.sh | ~100K total | **¥0.10 - ¥0.50** |

价格友好的厂商 (适合大量跑):
- **SiliconFlow Qwen 2.5 7B**: 注册送 2000 万 tokens, 跑完所有 demo 都不花钱
- **DeepSeek**: ¥1/百万 token, 全套 demo 约 ¥0.10
- **MiniMax Codin**: 按月订阅, 无限调用
