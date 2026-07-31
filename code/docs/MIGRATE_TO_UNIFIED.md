# 教程例子迁移到 UnifiedClient

> 核验日期：2026-07-31。`UnifiedClient` 统一的是仓库中的常见文本 Chat Completions 调用，
> 不是所有厂商 API、所有字段或所有模型能力的抽象。

## 1. 当前行为

| 条件 | 构造结果 | `chat()` 结果 |
|---|---|---|
| `LLM_MOCK=1` | 创建离线 client，不扫描/读取 `.env`、不返回/使用厂商 Key、不联网 | `resp.mock is True`，确定性本地响应 |
| `LLM_MOCK=0`，Key 与依赖齐全 | 创建真实客户端 | 成功时 `resp.mock is False` |
| `LLM_MOCK=0`，缺 Key | 构造时抛 `RuntimeError` | 不会自动进入 mock |
| 真实请求发生鉴权、网络或协议异常 | 已创建的真实客户端继续执行异常处理 | 返回 `resp.mock is True`、`model` 以 `error/` 开头，`raw` 保存异常 |

最后一行是“带错误标记的兜底响应”，不是成功。真实验收必须断言 `resp.mock is False`；
`quick_chat()` 只返回字符串，会隐藏这个状态，不适合作为真实 API 验收入口。

## 2. 标准迁移

### 历史 Before（仅展示迁移起点）

下面的 `gpt-4` 是旧示例中的硬编码模型：

```python
import openai

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
)
content = response.choices[0].message.content
```

### 当前写法

```python
import os

if os.environ.get("LLM_MOCK") != "0":
    print("[SKIP] 只有显式 LLM_MOCK=0 才执行真实调用")
    raise SystemExit(0)

from shared.llm_client import UnifiedClient

client = UnifiedClient()
resp = client.chat(
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=256,
)
if resp.mock:
    raise RuntimeError(f"真实 API 未通过：{type(resp.raw).__name__}: {resp.raw}")

print(f"[{resp.provider}/{resp.model}] {resp.content}")
print(resp.usage)
```

省略 `provider` 时，选择顺序由 `shared/provider_registry.py` 决定。生产或验收任务应显式指定，
避免机器上多个 Key 导致路由变化：

```python
client = UnifiedClient(provider="deepseek", model="deepseek-v4-flash")
```

## 3. 运行方式

离线运行：

```bash
LLM_MOCK=1 python your_example.py
```

真实运行：

```bash
LLM_MOCK=0 LLM_PROVIDER=deepseek python your_example.py
```

PowerShell：

```powershell
$env:LLM_MOCK="0"; $env:LLM_PROVIDER="deepseek"; python your_example.py
```

`UnifiedClient` 只把 `LLM_MOCK` 作为离线/真实模式开关；其他历史开关名称不会生效。

## 4. 能力边界

- `UnifiedClient.chat()` 会按 provider 的 `api_style` 路由：OpenAI-compatible 厂商使用
  `chat.completions.create()`，Anthropic 使用原生 `messages.create()`。
- Anthropic 统一层已覆盖字符串 system prompt、`user`/`assistant` 文本消息与 token usage 转换；
  tool/content blocks、多模态与扩展 thinking 参数仍应直接使用官方 SDK 或专用框架并单独测试。
- OpenAI `gpt-5.6*` 在当前实现中使用 `max_completion_tokens`，并按 reasoning 设置处理温度；
  其他厂商仍使用 `max_tokens`。这不代表其他 OpenAI 风格端点支持全部 OpenAI 参数。
- 多模态、Responses API、工具调用、严格 JSON Schema、缓存与思考内容均需专用适配和单独测试。
- 当前 provider、默认模型、base URL 与 Key 变量见 [API_KEYS.md](API_KEYS.md)。

## 5. Provider 选择注意

当前 provider 查找大小写不敏感；`UnifiedClient(provider="MiniMax")` 和
`LLM_PROVIDER=MiniMax` 都会命中 canonical name 为 `MiniMax` 的配置。未知名称会直接抛错，
不会静默切到 mock 或另一个已配置厂商；真实验收仍须核对 `resp.provider` 与目标厂商一致。

## 6. 真实验收

一个最小的严格探针：

```bash
LLM_MOCK=0 LLM_PROVIDER=deepseek python -c "from shared.llm_client import UnifiedClient; r=UnifiedClient().chat(prompt='Reply only OK', max_tokens=16); assert not r.mock, repr(r.raw); print(r.provider, r.model, r.usage)"
```

验收结果至少包含：

- 命令中的 `LLM_MOCK=0` 与显式 provider；
- `resp.mock is False`；
- 实际 provider/model、退出码和运行时间；
- 请求失败时的异常类型，而不是兜底文本；
- 目标字段或业务断言，而不只是“返回了非空字符串”。

离线 CI 与真实 API 验收要分开报告。离线通过只证明导入、分支和示例流程可运行：

```bash
LLM_MOCK=1 python -m pytest tests/ -m "not gpu" -q
```
