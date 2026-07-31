# LLM API Key 配置与验证

> 核验日期：2026-07-31。仓库仍同时存在 `UnifiedClient`、框架 factory 和厂商专用 SDK 示例，
> 不是所有 LLM 代码都能只靠一个 Key 无改动运行。

## 1. 当前 provider 注册表

`shared/provider_registry.py` 是仓库运行时配置的单一来源：

| provider | Key 环境变量 | 默认 chat | 默认 reasoner | base URL |
|---|---|---|---|---|
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` | `deepseek-v4-pro` | `https://api.deepseek.com` |
| `kimi` | `KIMI_API_KEY` | `kimi-k2.5` | `kimi-k2.5` | `https://api.moonshot.cn/v1` |
| `siliconflow` | `SILICONFLOW_API_KEY` | `Qwen/Qwen3.6-27B` | `deepseek-ai/DeepSeek-V4-Pro` | `https://api.siliconflow.cn/v1` |
| `MiniMax` | `MINIMAX_API_KEY` | `MiniMax-M2.7` | `MiniMax-M2.7` | `https://api.minimaxi.com/v1` |
| `openai` | `OPENAI_API_KEY` | `gpt-5.6` | — | `https://api.openai.com/v1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-fable-5` | — | `https://api.anthropic.com` |

这张表只表示仓库当前默认配置，不表示账户一定有权限，也不承诺价格、赠送权益、上下文、工具调用或
多模态能力。官方入口：

- [DeepSeek 开放平台](https://platform.deepseek.com/)与[模型/接口说明](https://api-docs.deepseek.com/quick_start/pricing/)
- [Kimi 开放平台](https://platform.moonshot.cn/)与[API 概述](https://platform.kimi.com/docs/api/overview)
- [SiliconFlow 控制台](https://cloud.siliconflow.cn/)与[模型列表接口](https://docs.siliconflow.cn/cn/api-reference/models/get-model-list)
- [MiniMax 开放平台](https://platform.minimaxi.com/)与[OpenAI 兼容说明](https://platform.minimaxi.com/docs/api-reference/text-openai-api)
- [OpenAI API 模型](https://developers.openai.com/api/docs/models)
- [Claude 模型概览](https://platform.claude.com/docs/en/about-claude/models/overview)

Kimi、SiliconFlow 等平台会动态调整模型。上线前查询 `/models` 或控制台，必要时通过 `model=...`
显式覆盖仓库默认值。

## 2. 配置

从 `code/` 目录复制模板：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中至少填写需要使用的一家，不要提交该文件：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=替换为真实密钥

KIMI_API_KEY=
SILICONFLOW_API_KEY=
MINIMAX_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

只有显式 `LLM_MOCK=0` 时，`shared.env` 才会读取进程 Key，或从 `code/` 及其父目录扫描并加载
`.env`。`LLM_MOCK=1` 和未设置 `LLM_MOCK` 都是离线模式：不扫描文件，也不返回或使用进程中已有
的厂商 Key。真实模式下系统环境变量优先级由 `python-dotenv` 默认行为决定。不要在代码、命令
历史、日志或截图中粘贴真实 Key。

## 3. 离线模式

未设置 `LLM_MOCK` 时默认离线；CI 应显式设置为 `1`，使日志和运行契约可审计：

```bash
LLM_MOCK=1 python -c "from shared.llm_client import UnifiedClient; c=UnifiedClient(); r=c.chat(prompt='hello'); assert c.is_mock and r.mock; print(r.model)"
```

PowerShell：

```powershell
$env:LLM_MOCK="1"; python -c "from shared.llm_client import UnifiedClient; c=UnifiedClient(); r=c.chat(prompt='hello'); assert c.is_mock and r.mock; print(r.model)"
```

这两种离线路径都在 provider 解析前短路，不扫描或读取 `.env`，不返回或使用厂商 Key，也不创建
网络客户端。

## 4. 真实调用与严格验证

真实调用必须显式设置 `LLM_MOCK=0`。先用 DeepSeek 示例做最小探针：

```bash
LLM_MOCK=0 LLM_PROVIDER=deepseek python -c "from shared.llm_client import UnifiedClient; r=UnifiedClient().chat(prompt='Reply only OK', max_tokens=16); assert not r.mock, repr(r.raw); print(r.provider, r.model, r.usage)"
```

PowerShell：

```powershell
$env:LLM_MOCK="0"; $env:LLM_PROVIDER="deepseek"; python -c "from shared.llm_client import UnifiedClient; r=UnifiedClient().chat(prompt='Reply only OK', max_tokens=16); assert not r.mock, repr(r.raw); print(r.provider, r.model, r.usage)"
```

也可运行诊断：

```bash
make llm-doctor-check PROVIDER=deepseek
# 等价的直接命令：
LLM_MOCK=0 LLM_PROVIDER=deepseek \
  python scripts/llm_doctor.py --provider deepseek --confirm-real
```

`--confirm-real` 会产生真实请求，并按失败数量返回非零退出码。自动化验收还应使用上面的
`assert not r.mock`，并增加业务断言。

### 标准代码

```python
import os

if os.environ.get("LLM_MOCK") != "0":
    raise SystemExit("[SKIP] 只有显式 LLM_MOCK=0 才执行真实调用")

from shared.llm_client import UnifiedClient

client = UnifiedClient(provider="deepseek")
resp = client.chat(prompt="用一句话解释 RAG", max_tokens=128)
if resp.mock:
    raise RuntimeError(f"真实调用失败：{type(resp.raw).__name__}: {resp.raw}")

print(f"[{resp.provider}/{resp.model}] {resp.content}")
print(resp.usage)
```

`quick_chat()` 只返回文本，无法检查 `resp.mock`，不要用它证明真实 API 已通过。

## 5. MiniMax 与 Anthropic 边界

- provider 查找大小写不敏感；`UnifiedClient(provider="MiniMax")`、
  `make_chat_model(provider="MiniMax")` 和 `LLM_PROVIDER=MiniMax` 会命中同一配置。
- MiniMax M2.7 的 OpenAI 兼容端点为 `https://api.minimaxi.com/v1`。部分 OpenAI 参数会被忽略或
  有不同约束，按[官方兼容说明](https://platform.minimaxi.com/docs/api-reference/text-openai-api)验收。
- Anthropic 在 `UnifiedClient.chat()` 中走原生 Messages API；当前统一层覆盖 system 字符串、
  `user`/`assistant` 文本消息和 usage 映射。工具调用、多模态 content blocks 与扩展 thinking
  参数使用 Anthropic 官方 SDK 或对应框架，并单独验收。

## 6. 常见失败

### 缺 Key

`LLM_MOCK=0` 下构造 `UnifiedClient` 会抛出明确错误，不会自动进入 mock。确认 Key 名、`.env` 位置，
以及启动进程是否加载了最新环境。

### 真实请求抛出异常

当前 `UnifiedClient` 对鉴权、模型权限、限流、网络或参数错误采用 fail-closed：保留底层异常并使
脚本失败，不再回退成 mock。旧日志若出现 `API ERROR, fallback to mock`，只能视为历史失败记录，
不能作为当前实现或实时 API 通过证据。

### factory 返回 `None`

`make_chat_model()` 在 `LLM_MOCK` 未设置、值不为 `0`，或显式厂商缺 Key 时返回 `None`。这表示
跳过，不是 mock 成功。

### SDK 缺失

```bash
python -m pip install -r requirements-llm.txt
```

## 7. 密钥安全

- 只把 `.env.example` 提交到版本库，不提交 `.env`。
- Key 仅放在服务端环境变量或密钥管理系统，禁止进入前端、Notebook 输出、日志和截图。
- 按厂商控制台提供的权限、预算和速率限制做最小授权；轮换与吊销周期遵循组织策略。
- 泄露后立即在厂商控制台吊销并重新签发，不要只删除 Git 中的明文。
