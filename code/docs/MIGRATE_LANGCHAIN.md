# LangChain / LlamaIndex 接入统一 ChatModel

> 核验日期：2026-07-31。本文只说明仓库当前 `shared/chatmodel_factory.py` 的行为。
> “OpenAI 兼容”表示可复用相应 SDK 的部分接口，不表示参数、响应字段和模型能力完全相同。

## 1. 先区分离线与真实调用

`make_chat_model()` 返回框架对象或 `None`，不会返回一个可调用的 Mock ChatModel：

- `LLM_MOCK=1`：直接返回 `None`，不扫描或读取 `.env`，不返回或使用厂商 Key，也不创建网络客户端。
- `LLM_MOCK=0` 且显式厂商缺 Key：返回 `None`。
- 未传 `provider` 且没有任何可用 Key：默认厂商解析会抛出 `RuntimeError`。
- 缺少 LangChain、LlamaIndex 或对应 Anthropic 集成包：抛出 `ImportError`。

因此，`None` 只能解释为“本次没有创建 ChatModel”，不能写成“已自动切换到 mock”，更不能计入
真实 API 通过数。

## 2. 标准迁移

### 历史 Before（仅展示迁移起点）

下面的 `gpt-4o-mini` 是旧示例中的硬编码模型，不是本教程的当前默认模型：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
chain = prompt | llm | parser
result = chain.invoke({"input": "hi"})
```

### 当前写法

```python
import os

if os.environ.get("LLM_MOCK") != "0":
    print("[SKIP] 只有显式 LLM_MOCK=0 才创建真实 ChatModel")
    raise SystemExit(0)

from shared.chatmodel_factory import make_chat_model

llm = make_chat_model(
    provider="deepseek",
    framework="langchain",
    temperature=0.7,
)
if llm is None:
    raise RuntimeError("未创建真实 ChatModel：请检查 LLM_MOCK 与 DEEPSEEK_API_KEY")

chain = prompt | llm | parser
result = chain.invoke({"input": "hi"})
print(result.content)
```

LlamaIndex 只需切换 `framework`：

```python
import os

if os.environ.get("LLM_MOCK") != "0":
    raise SystemExit("[SKIP] 离线模式")

from shared.chatmodel_factory import make_chat_model

llm = make_chat_model(
    provider="deepseek",
    framework="llama_index",
)
if llm is None:
    raise RuntimeError("未创建真实 LlamaIndex LLM")
```

真实运行必须显式关闭离线模式：

```bash
LLM_MOCK=0 LLM_PROVIDER=deepseek python ch18_llm_frameworks/llm/02_llmchain_basic.py
```

PowerShell：

```powershell
$env:LLM_MOCK="0"; $env:LLM_PROVIDER="deepseek"; python ch18_llm_frameworks/llm/02_llmchain_basic.py
```

离线检查则显式使用：

```bash
LLM_MOCK=1 python ch18_llm_frameworks/llm/02_llmchain_basic.py
```

## 3. 当前注册表映射

下表来自 `shared/provider_registry.py`，是仓库运行时默认值，不是厂商全部可用模型清单。

| provider 参数 | Key 环境变量 | 默认 chat / reasoner | base URL | factory 后端 |
|---|---|---|---|---|
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` / `deepseek-v4-pro` | `https://api.deepseek.com` | `ChatOpenAI` / `OpenAILike` |
| `kimi` | `KIMI_API_KEY` | `kimi-k2.5` / `kimi-k2.5` | `https://api.moonshot.cn/v1` | `ChatOpenAI` / `OpenAILike` |
| `siliconflow` | `SILICONFLOW_API_KEY` | `Qwen/Qwen3.6-27B` / `deepseek-ai/DeepSeek-V4-Pro` | `https://api.siliconflow.cn/v1` | `ChatOpenAI` / `OpenAILike` |
| `MiniMax` | `MINIMAX_API_KEY` | `MiniMax-M2.7` / `MiniMax-M2.7` | `https://api.minimaxi.com/v1` | `ChatOpenAI` / `OpenAILike` |
| `openai` | `OPENAI_API_KEY` | `gpt-5.6` / — | `https://api.openai.com/v1` | `ChatOpenAI` / `OpenAILike` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-fable-5` / — | `https://api.anthropic.com` | `ChatAnthropic` / `Anthropic` |

权威接口与模型入口：

- [DeepSeek Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [Kimi API 概述](https://platform.kimi.com/docs/api/overview)与[模型列表](https://platform.kimi.com/docs/models)
- [SiliconFlow 获取模型列表](https://docs.siliconflow.cn/cn/api-reference/models/get-model-list)
- [MiniMax OpenAI API 兼容](https://platform.minimaxi.com/docs/api-reference/text-openai-api)
- [OpenAI Models](https://developers.openai.com/api/docs/models)
- [Claude 模型概览](https://platform.claude.com/docs/en/about-claude/models/overview)

模型会动态上下线。真实运行前可用厂商 `/models` 接口或控制台确认账户实际可用模型；不要从这张表
推导价格、赠送权益、上下文长度或特定高级参数一定受支持。

### Provider 名称

当前注册表查找大小写不敏感，并保留 canonical name；`provider="MiniMax"` 与
`LLM_PROVIDER=MiniMax` 都会命中同一配置。真实验收仍应打印实际 provider 与 model，避免拼写错误
被未知 provider 的 mock 配置掩盖。

## 4. 兼容性边界

- DeepSeek、Kimi、SiliconFlow、MiniMax 和 OpenAI 在 factory 中走 OpenAI 风格的客户端。
- Anthropic 走独立的 LangChain/LlamaIndex Anthropic 集成，不经过 `ChatOpenAI`。
- LlamaIndex `OpenAILike` 的 `context_window` 默认值是 factory 的配置参数，不是对所有厂商真实能力的承诺。
- 工具调用、JSON Schema、缓存、思考字段、多模态和流式事件必须按“厂商 + 模型 + SDK 版本”单独验收。
- `make_openai_client()` 只有在 `LLM_MOCK=0` 时才允许建真实客户端；未设置或其他值均会抛错。

## 5. 验收清单

真实验收至少记录：

1. `LLM_MOCK=0`、显式 provider、实际返回的模型名与运行时间；
2. 依赖版本、请求参数、退出码和完整错误类型；
3. 至少一个正常响应，以及目标功能对应的工具调用或结构化输出断言；
4. `None`、`[SKIP]`、依赖缺失和离线运行不得计入真实通过。

离线 CI 只验证代码路径和确定性跳过：

```bash
LLM_MOCK=1 python -m pytest tests/ -m "not gpu" -q
```
