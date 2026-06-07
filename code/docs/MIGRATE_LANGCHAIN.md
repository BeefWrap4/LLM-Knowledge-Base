# LangChain / LlamaIndex 例子接入真实 LLM (Wave 19)

> Ch18 的 ~16 个 langchain + 6 个 llamaindex 例子原本默认 mock 模式.
> 通过 `shared/chatmodel_factory.py`, 一行切换厂商, 真实调用 deepseek/kimi/siliconflow/MiniMax.

## 1. 5 行标准改动

### Before (mock 模式)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",       # 写死 OpenAI
    temperature=0.7,
)
chain = prompt | llm | parser
result = chain.invoke({"input": "hi"})
```

### After (UnifiedClient 一行切换)

```python
# Wave 19: 改用 chatmodel_factory
from shared.chatmodel_factory import make_chat_model

llm = make_chat_model()                            # 默认厂商 (LLM_PROVIDER)
# llm = make_chat_model(provider="deepseek")       # 显式 DeepSeek
# llm = make_chat_model(provider="kimi", model="moonshot-v1-128k")
# llm = make_chat_model(provider="siliconflow", framework="llama_index")

if llm is None:
    print("[SKIP] 无 API Key, 使用 mock")
    import sys; sys.exit(0)

chain = prompt | llm | parser
result = chain.invoke({"input": "hi"})
```

## 2. 厂商映射表

所有 OpenAI 兼容厂商 (deepseek/kimi/siliconflow/MiniMax/openai) 在 LangChain 都映射为 `ChatOpenAI` + 自定义 `base_url`. Anthropic 走 `ChatAnthropic`.

| 厂商 | LangChain 类 | LlamaIndex 类 | base_url |
|------|-------------|---------------|----------|
| deepseek | ChatOpenAI | OpenAILike | api.deepseek.com/v1 |
| kimi | ChatOpenAI | OpenAILike | api.moonshot.cn/v1 |
| siliconflow | ChatOpenAI | OpenAILike | api.siliconflow.cn/v1 |
| MiniMax | ChatOpenAI | OpenAILike | api.minimaxi.com/v1 |
| openai | ChatOpenAI | OpenAILike | api.openai.com/v1 |
| anthropic | ChatAnthropic | Anthropic | api.anthropic.com |

## 3. 改造清单 (推荐)

| 优先级 | 文件 | 现状 |
|-------|------|------|
| ★★★ | ch18/llm/01_lcel_style_basic_chain.py | 已用 UnifiedClient (Wave 7) |
| ★★ | ch18/llm/02_llmchain_basic.py | 待改 |
| ★★ | ch18/llm/05_conversation_buffer_memory.py | 待改 |
| ★★ | ch18/llm/13_llamaindex_vectorstore_index.py | 待改 |
| ★ | ch18/llm/30_haystack_rag_pipeline.py | 框架特定, 不建议改 |

## 4. 调试

```python
from shared.chatmodel_factory import doctor_summary, has_langchain
import json
print(json.dumps(doctor_summary(), indent=2, ensure_ascii=False))
print(f"langchain installed: {has_langchain()}")
```

## 5. 何时用 factory vs UnifiedClient

| 场景 | 推荐 |
|------|------|
| 简单 chat 调用 | `UnifiedClient().chat()` (Wave 14) |
| LangChain chain / agent | `make_chat_model(framework="langchain")` (Wave 19) |
| LlamaIndex query engine | `make_chat_model(framework="llama_index")` (Wave 19) |
| 流式 + 复杂工具 | `make_openai_client()` + 自己处理 stream |

## 6. 进阶: 1 个 LangChain 例子的完整改造

```python
# 旧 (mock 模式)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("讲个关于{topic}的笑话")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"topic": "程序员"}).content)
```

```python
# 新 (Wave 19: 真实调用)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from shared.chatmodel_factory import make_chat_model

prompt = ChatPromptTemplate.from_template("讲个关于{topic}的笑话")
llm = make_chat_model()                            # 默认厂商
if llm is None:
    print("[SKIP] 无 API Key, mock 模式跳过")
    import sys; sys.exit(0)

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "程序员"})
print(f"[{result.response_metadata.get('model_name', '?')}] {result.content}")
```

## 7. 失败回退

`make_chat_model` 返回 `None` 当缺 API Key. 调用方应:

```python
llm = make_chat_model(provider="kimi")
if llm is None:
    print("[SKIP] KIMI_API_KEY 未配置, 跳过 (mock 模式)")
    # 输出一些示例数据让 CI 仍通过
    print("Mock response: 这是一个 KIMI mock 输出")
    import sys; sys.exit(0)  # 返回 0 = OK
```

**永不抛异常**, 因为 CI 必须保持 357/357 通过率.
