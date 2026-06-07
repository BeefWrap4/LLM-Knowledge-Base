# LLM API Key 配置指南

> **5 分钟接入真实 LLM, 0 行业务改造**. 本教程所有调用 LLM 的代码已统一通过 `shared/llm_client.py`, 您只需在 `.env` 文件中填入 1 个 API Key 即可让本地代码真正调用云端大模型.

## 1. 快速开始 (3 步)

### Step 1: 复制环境变量模板

```bash
cd code/
cp .env.example .env
```

### Step 2: 注册并获取 API Key (选 1 家, 推荐 DeepSeek 或 Kimi)

| 厂商 | 注册地址 | 免费额度 | 推荐场景 |
|------|---------|---------|---------|
| **DeepSeek** | https://platform.deepseek.com | 注册送 ¥10 | 强推理, R1 模型, ¥1/百万 token |
| **Kimi (月之暗面)** | https://platform.moonshot.cn | 新用户 ¥15 体验金 | 长上下文 128K, 中文友好 |
| **SiliconFlow** | https://cloud.siliconflow.cn | 注册送 2000 万 tokens | 多模型路由, 性价比高 |
| OpenAI | https://platform.openai.com | 需信用卡 | GPT-4o-mini 通用 |

### Step 3: 填入 `.env` 并验证

```bash
# .env 文件 (至少填 1 个)
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

```bash
# 验证 Key 是否有效
cd code/
python scripts/llm_doctor.py
```

**期望输出**:
```
[✓] deepseek     (CN)  OK                       1.15s
[✓] kimi         (CN)  OK                       0.5s
[✓] siliconflow  (CN)  OK                       0.4s
Result: 3 passed, 0 failed
```

## 2. 在代码中使用

### 最简调用 (1 行)

```python
from shared.llm_client import quick_chat

answer = quick_chat("用一句话解释量子计算")
print(answer)
```

### 标准调用 (可控制模型/温度)

```python
from shared.llm_client import UnifiedClient

client = UnifiedClient()                            # 默认厂商 (LLM_PROVIDER)
# client = UnifiedClient(provider="kimi")           # 显式指定
# client = UnifiedClient(provider="siliconflow", model="Qwen/Qwen2.5-72B-Instruct")

resp = client.chat(
    prompt="分析以下评论的情感: 这家餐厅太棒了!",
    system="你是情感分析专家, 输出 '正面/负面/中性'",
    temperature=0.3,
    max_tokens=50,
)

print(f"[{resp.provider}/{resp.model}] {resp.content}")
print(f"  tokens: {resp.usage}")          # 真实 token 统计
print(f"  mock: {resp.mock}")              # False = 真实调用
```

### 流式调用 (打字机效果)

```python
from openai import OpenAI
from shared.provider_registry import get_provider

p = get_provider("deepseek")
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=p.base_url)

stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一首关于编程的诗"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## 3. 厂商深度对比

### DeepSeek

| 模型 | 上下文 | 用途 | 价格 |
|------|--------|------|------|
| `deepseek-chat` | 32K | V3 通用对话 | ¥1/百万 input, ¥2/百万 output |
| `deepseek-reasoner` | 32K | R1 推理 (CoT 可见) | ¥4/百万 input, ¥16/百万 output |

- 优势: 推理能力强 (R1), 国内访问快, OpenAI 兼容
- 适用: Ch27 reasoning, Ch15 ReAct agent, 任何需要 Chain-of-Thought 的场景
- 注册: 微信/邮箱均可, 实名 5 分钟

### Kimi (Moonshot)

| 模型 | 上下文 | 用途 | 价格 |
|------|--------|------|------|
| `moonshot-v1-8k` | 8K | 短对话 | ¥12/百万 tokens |
| `moonshot-v1-32k` | 32K | 中等 | 同上 |
| `moonshot-v1-128k` | 128K | 长文档 | ¥60/百万 tokens |

- 优势: 128K 长上下文, 中文 PDF 解析强
- 适用: Ch14 RAG 长文档, Ch22 数据清洗, Ch29 上下文工程

### SiliconFlow

| 模型 | 用途 |
|------|------|
| `Qwen/Qwen2.5-7B-Instruct` | 阿里通义 7B |
| `Qwen/Qwen2.5-72B-Instruct` | 阿里通义 72B |
| `Qwen/QwQ-32B-Preview` | 推理 |
| `deepseek-ai/DeepSeek-V3` | DeepSeek V3 (SiliconFlow 镜像) |
| `THUDM/glm-4-9b-chat` | 智谱 GLM-4 |

- 优势: 单一 API key 访问多个开源模型, 价格低 (部分免费)
- 适用: 想用 Qwen/GLM/DeepSeek 但不想注册多家

## 4. 故障排查

### `ModuleNotFoundError: No module named 'openai'`

```bash
pip install -r requirements-llm.txt
```

### `[WARN] 无 DEEPSEEK_API_KEY, 降级到 MockLLM`

说明 `.env` 未生效. 检查:
1. 文件名是 `.env` 不是 `.env.example` (后者是模板)
2. 文件在 `code/` 目录下
3. 格式是 `KEY=value` 不是 `KEY: value` (yaml 风格会失败)
4. 重启 Python 进程 (env 在启动时加载)

### `[API ERROR] Invalid API key`

- 检查 Key 是否完整复制 (没有空格/换行)
- 登录厂商后台确认 Key 未过期
- DeepSeek 偶尔需要重新生成

### Kimi 返回 `[API ERROR, fallback to mock]`

Kimi 的 Moonshot API 偶尔 rate-limit. 切换到 DeepSeek:
```bash
LLM_PROVIDER=deepseek python your_script.py
```

### 想清空所有 Key 测试 mock

```bash
unset DEEPSEEK_API_KEY KIMI_API_KEY SILICONFLOW_API_KEY
python code/scripts/llm_doctor.py
# → 全部降级到 mock
```

## 5. 安全提示

- **永远不要** 提交 `.env` 到 git (已在 `.gitignore`)
- **永远不要** 在代码中硬编码 API Key
- 定期轮换 Key (建议每 90 天)
- 为每个项目创建独立的 Key (厂商支持多 Key 限额隔离)
