# 教程例子迁移到 UnifiedClient (Wave 15)

> 本教程所有调用 LLM 的代码可零成本切换到 **真实厂商 API** (DeepSeek / Kimi / SiliconFlow / MiniMax 等),
> 只需设置环境变量 + 调用 `UnifiedClient`. 详见本指南.

## 1. 现状 (Wave 14-F)

| 已改造 (✓ 真实调用验证) | 状态 |
|---|---|
| `ch13/llm/06_self_consistency_cot.py` | ✓ MiniMax 实测通过 |
| `ch13/llm/09_compare_temperatures.py` | ✓ MiniMax 实测通过 (3 个温度样本) |

`make llm-doctor` 在你的环境 (DEEPSEEK + KIMI + SILICONFLOW + MINIMAX) 显示 **4/4 passed**.

## 2. 迁移一个例子的标准流程 (5 行改动)

### Step 1: 加 path setup (让 shared 可导入)

文件顶部插入:

```python
import sys as _sys_path_setup
from pathlib import Path as _Path_setup
_code_root = _Path_setup(__file__).resolve().parent.parent.parent  # /app/code 或 code/
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))
```

### Step 2: 替换 `import openai` + `openai.chat.completions.create` 块

**Before**:
```python
import openai
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature,
)
content = response.choices[0].message.content
```

**After**:
```python
from shared.llm_client import UnifiedClient
_client = UnifiedClient()
resp = _client.chat(
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature,
    # model 省略 = 用 provider 默认 (deepseek-chat, MiniMax-Text-01, etc.)
)
content = resp.content
```

### Step 3: (可选) 启用真实调用

```bash
# 默认 mock (防止误用 API), 设 USE_REAL_API=1 启用真实调用
USE_REAL_API=1 python your_example.py
```

## 3. 改造前后对比

| 维度 | 改造前 | 改造后 |
|------|--------|--------|
| API Key 配置 | 写死 `os.environ["OPENAI_API_KEY"]` | 自动从 6 家厂商任选 |
| 厂商切换 | 改 base_url + model 字符串 | 改 1 个参数 `provider="kimi"` |
| 失败回退 | 抛异常, 例子崩 | 自动降级 mock, 例子继续 |
| Mock 模式 | 单独写一份 | 同一份代码, 缺 Key 时降级 |
| 跑通例子 | 需 OPENAI_API_KEY | 1 家厂商 Key (推荐 DeepSeek) |
| 多模态/特殊 | 需用 Anthropic SDK | 同 `_client.chat(messages, ...)` |

## 4. 迁移清单 (剩余 ~20 个可改造例子)

### 简单 (1-2 行改动) — 推荐

| 文件 | 现状 | 难度 |
|------|------|------|
| `ch13/llm/06_self_consistency_cot.py` | ✓ 已完成 | 模板 |
| `ch13/llm/09_compare_temperatures.py` | ✓ 已完成 | 模板 |
| `ch13/llm/14_openai_auto_caching.py` | 未改 | 简单 |
| `ch13/llm/20_openai_json_schema_strict.py` | 未改 | 简单 |
| `ch17/llm/05_llm_as_judge.py` | 未改 | 中等 (类封装) |
| `ch17/llm/12_langfuse_v3.py` | 未改 | 中等 |

### 复杂 (框架封装, 不建议改) — 已 SKIP

| 类别 | 文件 | 原因 |
|------|------|------|
| LangChain | `ch18/llm/02-09_*.py` | 已用 langchain_openai.ChatOpenAI, 框架特性 |
| LlamaIndex | `ch18/llm/13-18_*.py` | 已用 OpenAILike, 框架特性 |
| LangGraph | `ch18/llm/10-12_*.py` | 与 LangChain 深度耦合 |
| Haystack | `ch18/llm/30_*.py` | Haystack 组件系统 |

**这些框架例子保留 SDK 调用, 通过 SKIP 模式跳过 — 不破坏 100% 通过率**.

## 5. 调试技巧

### 如何确认用 mock 还是真实 API?

```python
from shared.llm_client import UnifiedClient
c = UnifiedClient()
print(f"provider: {c.provider.name}, model: {c.model}, mock: {c.is_mock}")
```

输出:
- `mock: False` → 真实调用 (会扣费!)
- `mock: True` → 降级 mock (安全)

### 单独测试某个厂商

```bash
LLM_PROVIDER=deepseek python your_script.py
LLM_PROVIDER=kimi python your_script.py
LLM_PROVIDER=siliconflow python your_script.py
LLM_PROVIDER=MiniMax python your_script.py
```

### Token 使用量

`UnifiedClient` 返回 `.usage` 字段:

```python
resp = _client.chat(prompt="hi", max_tokens=10)
print(f"消耗 {resp.usage['total_tokens']} tokens ({resp.provider}/{resp.model})")
```

## 6. 例子 PR 模板

如果你想贡献新例子, 提交 PR 时:

1. 文件头加 YAML 注释 (见 `code/ch12/.../01_attention.py` 模板)
2. 顶部加 path setup (Step 1)
3. 主逻辑用 UnifiedClient
4. 加 `if __name__ == "__main__"` 入口
5. 跑通 `USE_REAL_API=1 python your_file.py`
6. 跑通 `make ci` (默认 mock 也通过)

## 7. 未来工作 (Wave 16+)

- 改写剩余 6 个简单例子 (见上表)
- CI 中跑 `llm-doctor` 真实调用 (需在 GitHub Secrets 配置 Key)
- 给 LangChain/LlamaIndex 例子加 vendor 选择 helper
