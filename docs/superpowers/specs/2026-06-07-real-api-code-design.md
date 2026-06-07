# Real API / Real Model / Real Framework — 配套代码真实化设计

| | |
|--|--|
| **日期** | 2026-06-07 |
| **作者** | Claude (brainstorming session) |
| **状态** | 设计草案，等待用户审查 |
| **目标范围** | `code/` 450 个 .py + 教程 29 个 .md + CI 脚本 |

---

## 0. 背景与目标

**问题**：`code/` 目录的 450 个 .py 文件中，29+ 文件直接 import `MockLLM`，199 个 LLM tier 文件有 `is_mock / USE_REAL_API / provider == "mock"` 等条件分支；76 个 GPU tier 文件中部分（如 `ch25/08_tensorrt_llm_build_mock.py`、`ch25/10_vllm_async_engine_client.py`）是**完全自实现的 mock 类**，从未连接真实引擎。结果：用户 clone 后 `python xx.py` 经常得到"这是个 mock 响应"而不是真实输出。

**目标**：教程配套代码全部使用真实 API、真实模型（缺失权重则下载）、真实框架；主流程不出现 mock；mock 路径下沉为 CI 专用。

**6 个核心决定**（用户已确认）：
1. **目标**：教程交付物导向——主流程无 mock
2. **GPU tier**：本机能跑就跑，缺硬件给"硬件需求"提示而非降级 mock
3. **CI / mock**：主代码保留 `LLM_MOCK=1` 单行开关，CI 用环境变量触发
4. **权重下载**：显式 `make download-models --target=xxx`
5. **默认 API**：DeepSeek-V3 主推（OpenAI 协议 + 注册送 ¥10）
6. **教程同步**：教程 markdown 与代码完全同步更新

---

## 1. 整体架构（4 层）

```
┌─────────────────────────────────────────────────────────────────────┐
│ L1 教程层 (29 × .md)                                                │
│   - Ch18/25/26/28 重灾章节实质性重写                                 │
│   - 其他章节替换"运行命令"段                                         │
│   - 引用 README 硬件矩阵                                             │
└─────────────────────────────────────────────────────────────────────┘
                              │ 反向引用
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L2 代码层 (450 × .py)                                               │
│   - 主流程: 默认走真实 API / 真实模型 / 真实框架                      │
│   - 残留: 仅 1 个环境变量开关 `LLM_MOCK=1` (A2 方案)                 │
└─────────────────────────────────────────────────────────────────────┘
                              │ 使用
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L3 共享层 (shared/)                                                 │
│   - UnifiedClient:    真实 OpenAI 协议客户端，缺 Key 抛错            │
│   - provider_registry: 6 个厂商 + deepseek 标记为推荐默认            │
│   - chatmodel_factory: LangChain / LlamaIndex 统一工厂              │
│   - mock_llm.py:      从 shared/ 下沉到 tests/_mocks/                │
└─────────────────────────────────────────────────────────────────────┘
                              │ 调用
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L4 外部服务                                                          │
│   - DeepSeek / Kimi / SiliconFlow / MiniMax / OpenAI / Anthropic    │
│   - Ollama (本地 OpenAI 协议, base_url=http://localhost:11434/v1)   │
│   - vLLM / SGLang / TRT-LLM (本地 OpenAI 协议 server)               │
│   - 本地模型权重 (code/models/, ModelScope / hf-mirror 下载)         │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键不变量

- **L2 主流程不出现 `is_mock` / `provider == "mock"` 分支**（违反一行 = PR 拒绝）
- **L2 仅允许 1 行环境变量开关**：`if os.environ.get("LLM_MOCK") == "1": ...`，且必须紧跟 `sys.exit(0)`（不能跑真实路径）
- **L3 `shared/mock_llm.py` 在 W3 迁移到 `tests/_mocks/`**，并从 `shared/__init__.py` 移除导出
- **L4 网络/硬件缺失时 L3 抛明确异常**（"需要 DEEPSEEK_API_KEY"、"需要 NVIDIA 24GB+ GPU"），不静默降级

---

## 2. 核心组件改造清单

### 2.1 `shared/llm_client.py` — `UnifiedClient` 行为反转

**当前**：缺 Key → 打印 `[WARN]` → 降级 `MockLLM`  
**目标**：缺 Key → 抛 `RuntimeError("需要 DEEPSEEK_API_KEY，参考 README §硬件矩阵")`

```python
# 改造后
class UnifiedClient:
    def __init__(self, ...):
        ...
        if not self.api_key or self.api_key == "YOUR_API_KEY":
            raise RuntimeError(
                f"厂商 {self.provider.name} 缺 API Key (env {self.provider.env_key}). "
                f"运行 `make llm-doctor` 诊断；或 `export LLM_MOCK=1` 使用 mock (仅测试)."
            )
```

**CI 用法**：`LLM_MOCK=1 python xx.py` 触发 mock；用户开箱 `python xx.py` 走真实。  
**改动量**：~30 行

### 2.2 `shared/provider_registry.py` — 默认优先级

`get_default_provider()` 当前：CN 第一个有 Key → US 第一个有 Key → mock  
**改为**：CN 第一个有 Key → US 第一个有 Key → **抛 `RuntimeError`**

**改动量**：~15 行

### 2.3 `shared/mock_llm.py` — 整体下移到 `tests/_mocks/`

| 当前位置 | 目标位置 |
|----------|---------|
| `shared/mock_llm.py` | `tests/_mocks/mock_llm.py` |
| `shared/__init__.py` 导出 | 移除 |
| 引用它的 29 个 .py | 测试入口改到 `tests/_mocks/` |

### 2.4 受影响文件清单（29+ 个 .py）

| 章节 | 文件数 | 改造方式 |
|------|-------|---------|
| `ch18_llm_frameworks/llm/01-09_*.py` | 9 | 删除 `run_real()/run_mock()` 双函数；只留真实路径 |
| `ch18_llm_frameworks/llm/13-18_*.py` | 6 | 删除 `if USE_REAL_API:` 分支，保留真实路径 |
| `ch18_llm_frameworks/llm/35_memory_token_control.py` | 1 | 同上 |
| `ch19_distributed/gpu/02_ddp_training.py` | 1 | W3 阶段：仅审 + 去 mock import；W6 阶段：真实 `accelerate launch`（见 §6 W6）|
| `ch22_data_eng/llm/04_self_instruct.py` | 1 | 删 `if not is_real_api` |
| `ch22_data_eng/llm/11_constitutional_ai.py` | 1 | 同上 |
| `ch29_context_engineering/llm/05_haystack_chat_pipeline.py` | 1 | 同上 |
| `ch29_context_engineering/llm/12_full_context_pipeline.py` | 1 | 同上 |
| `ch25_inference_engines/gpu/08_tensorrt_llm_build_mock.py` | 1 | **改名** `08_tensorrt_llm_build.py`，真实 `trtllm-build` |
| `ch25_inference_engines/gpu/10_vllm_async_engine_client.py` | 1 | **改名** `10_vllm_async_engine.py`，真实 `vllm.AsyncLLMEngine` |
| 其他 `ch25/ch26/ch28` 自实现 mock 类 | ~12 | 同上 |

### 2.5 `scripts/llm_doctor.py` — 升级为引导式

新增 `--setup` 模式：
- 检测无 Key → 提示 `deepseek` 注册链接 → 生成 `.env` 模板
- 检测到 Key → 测试一次最小调用（10 token）→ 报告延迟/成功率
- 检测到 `ollama serve` → 自动加入可用厂商

### 2.6 `scripts/download_models.py` — 扩到全清单

从当前 3 个模型扩到 12+：

| 章节 | 模型 | 大小 | 用途 |
|------|------|------|------|
| Ch14/17/20/22 | bge-small-zh-v1.5 | 100MB | embedding（已有）|
| Ch17/22 | bge-reranker-v2-m3 | 600MB | reranker（已有）|
| Ch12-18 | Qwen2.5-0.5B-Instruct | 1GB | 通用 LLM 小（已有）|
| Ch25 | Qwen2.5-7B-Instruct | 15GB | vLLM 真实启动（新增）|
| Ch25 | Llama-3.1-8B-Instruct | 16GB | vLLM 备用（新增）|
| Ch26 | Cosmos-1.0-7B | 14GB | 世界模型 demo（新增）|
| Ch26 | Pi0-VLA-base | 8GB | VLA demo（新增）|
| Ch27 | DeepSeek-R1-Distill-Qwen-1.5B | 3GB | 推理模型 demo（新增）|
| Ch28 | mlx-community/Qwen2.5-7B-Instruct-4bit | 5GB | Apple Silicon MLX（新增）|
| Ch28 | llama.cpp/llama-3.2-3b-instruct-q4_k_m | 2GB | llama.cpp（新增）|
| Ch19 | Qwen2.5-0.5B + DDP sharded | 1GB×2 | DDP/FSDP 训练（复用）|
| Ch16 | Qwen2.5-0.5B + LoRA adapters | 1.1GB | SFT/LoRA（复用）|

**总下载量约 65GB**（按需选；README 矩阵告诉每章需要哪些）。

---

## 3. 数据流（真实调用流程）

### 3.1 LLM tier 调用流（以 `ch18/01_langchain_basic_chain.py` 为例）

```
用户执行: python ch18_llm_frameworks/llm/01_langchain_basic_chain.py
                    │
                    ▼
   ┌─────────────────────────────────────┐
   │ UnifiedClient()                     │  ← shared/llm_client.py
   │   - 读 env DEEPSEEK_API_KEY         │
   │   - 选默认厂商 (deepseek)           │
   │   - 无 Key → 抛 RuntimeError        │
   └─────────────────────────────────────┘
                    │ 有 Key
                    ▼
   ┌─────────────────────────────────────┐
   │ OpenAI(                             │  ← openai SDK
   │   api_key=DEEPSEEK_API_KEY,         │
   │   base_url="https://api.deepseek.com/v1"
   │ )                                   │
   └─────────────────────────────────────┘
                    │ HTTPS POST
                    ▼
   ┌─────────────────────────────────────┐
   │ DeepSeek API                        │  ← 真实服务端
   │   model=deepseek-chat (V3)          │
   └─────────────────────────────────────┘
                    │ 响应
                    ▼
   ┌─────────────────────────────────────┐
   │ LLMResponse                         │  ← 统一返回对象
   │   .content / .usage / .raw          │
   └─────────────────────────────────────┘
```

**CI 路径**：`LLM_MOCK=1 python xx.py` → 跳过上面所有步骤，直接走 `tests/_mocks/mock_llm.py:MockLLM`。

### 3.2 LlamaIndex 路径（`ch18/13_llamaindex_vectorstore_index.py`）

```
Settings.llm = make_chat_model(framework="llama_index")
                │
                ├─ provider=deepseek (有 Key)
                │    └─ OpenAILike(model="deepseek-chat", base_url=deepseek_url)
                │
                └─ 无 Key
                     └─ 抛 RuntimeError (主流程)
                     └─ LLM_MOCK=1 → tests/_mocks 里的 MockLLM

Settings.embed_model = HuggingFaceEmbedding("code/models/bge-small-zh-v1.5")
                │
                ├─ 权重已下载 → 本地 CPU/GPU 加载 bge
                │
                └─ 权重缺失 → 抛 RuntimeError("需要 make download-models --target=embedding")
```

### 3.3 GPU tier — vLLM 真实启动（`ch25/10_vllm_async_engine.py`）

```python
from vllm import AsyncLLMEngine as VLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams as VPSamplingParams

async def main():
    args = AsyncEngineArgs(
        model="code/models/Qwen2.5-7B-Instruct",
        max_num_seqs=64,
        gpu_memory_utilization=0.9,
    )
    engine = VLLMEngine.from_engine_args(args)
    sampling = VPSamplingParams(temperature=0.7, max_tokens=64)
    async for out in engine.generate("讲个笑话", sampling, request_id="r-1"):
        print(out.text, end="", flush=True)
```

**前置检查**：
- `torch.cuda.is_available()` 否则 `raise RuntimeError("需要 NVIDIA GPU")`
- `Path("code/models/Qwen2.5-7B-Instruct/config.json").exists()` 否则 `raise RuntimeError("需要 make download-models --target=qwen7b")`

### 3.4 GPU tier — Ollama 端侧（`ch28/05_ollama_http_api.py`）

```python
import httpx
try:
    resp = httpx.post(
        "http://localhost:11434/api/chat",
        json={"model": "llama3.2:3b",
              "messages": [{"role": "user", "content": "Hello!"}],
              "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    print(resp.json()["message"]["content"])
except httpx.ConnectError:
    raise RuntimeError("Ollama 未运行. 先 `ollama serve` + `ollama pull llama3.2:3b`")
```

### 3.5 DDP/FSDP 真实训练（`ch19/02_ddp_training.py`）

```python
if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("DDP 需要 ≥2 张 NVIDIA GPU")
    # subprocess: accelerate launch --num_processes 2 02_ddp_training.py
```

---

## 4. 错误处理（统一异常体系）

### 4.1 三类错误的统一出口

| 错误类型 | 异常类 | 信息模板 |
|---------|--------|---------|
| **A. 缺 API Key** | `RuntimeError` | `厂商 {provider} 缺 API Key (env {env_var})。运行 \`make llm-doctor\` 诊断。` |
| **B. 缺模型权重** | `RuntimeError` | `需要模型权重 {path}。运行 \`make download-models --target={target}\`。` |
| **C. 缺硬件** | `RuntimeError` | `此例子需要 {hardware}（{detail}）。详见 README §"硬件 × 章节矩阵"。` |
| **D. 网络故障** | 透传 | `openai.APIConnectionError` 等 |

### 4.2 友好提示的固定格式

```
[ERROR] {file}:{line}  {异常类}: {消息}
[HELP]  参考 README#硬件矩阵 或运行 `make llm-doctor --setup`
[HELP]  或 `export LLM_MOCK=1` 用 mock 跑 (仅 CI/离线)
```

通过 `shared/_error_helper.py`（新增 ~40 行）统一输出。

### 4.3 错误处理的边界规则

- ❌ 不在 catch 里降级到 mock（违反 A2 方案核心）
- ❌ 不在 catch 里静默重试（让用户立即看到失败）
- ✅ Network/timeout 错误透传（真实环境问题，不应被掩盖）

### 4.4 GPU 硬件检测的统一入口

`shared/gpu_guard.py` 新增：

```python
def require_nvidia_gpu(min_vram_gb: int = 8, min_count: int = 1) -> None:
    """检查 NVIDIA GPU + 显存 + 数量. 不足抛 RuntimeError."""

def require_apple_silicon(min_memory_gb: int = 8) -> None:
    """检查 Apple Silicon. 否则抛 RuntimeError."""

def require_ollama(model: str = "llama3.2:3b") -> None:
    """检查 Ollama 服务 + 模型. 否则抛 RuntimeError."""
```

### 4.5 错误处理的反面教材（不做什么）

- ❌ 缺 Key 时降级到 mock
- ❌ 缺 GPU 时用 CPU 跑（性能差会误导）
- ❌ 把异常吞掉打印"略"
- ❌ 默认 5 次重试

---

## 5. 测试与 CI 改造

### 5.1 双轨 CI

**PR 检查（≤3 分钟）**：
1. `verify_all.py`（wiki/链接/章节）—— 10s
2. `pytest tests/ -m "not gpu"`（mock 冒烟）—— 60s
3. `ruff + mypy` 类型检查 —— 30s
4. `llm_doctor.py` Key 状态报告 —— 5s

**夜间（22:00 UTC）跑真实集成测试**：
1. 真实 API 冒烟（`DEEPSEEK_API_KEY`）—— 3min（10 个代表性文件）
2. 真实 GPU 冒烟（self-hosted runner）—— 10min（5 个 vLLM/Ollama 例子）
3. 真实权重完整性校验（`code/models/*/config.json` sha256）

### 5.2 `tests/_mocks/` 新结构

```
tests/
├── conftest.py                # pytest fixtures
├── test_pilots.py             # 现有 smoke tests
├── test_shared.py             # 现有
├── _mocks/                    # 新增 — 仅 CI 可见
│   ├── __init__.py            # 暴露 MockLLM, deterministic_response
│   ├── mock_llm.py            # ← 从 shared/ 移过来
│   ├── mock_embedding.py      # ← ch18 13-18 里 _MockEmbed 抽出来
│   ├── mock_vllm.py           # ← ch25/10 移过来
│   ├── mock_trt.py            # ← ch25/08 移过来
│   └── conftest.py            # 自动设 LLM_MOCK=1
└── test_real_api_smoke.py     # 夜间跑，标记 @pytest.mark.nightly
```

### 5.3 `LLM_MOCK=1` 行为矩阵

| 场景 | LLM_MOCK 未设 | LLM_MOCK=1 |
|------|-------------|-----------|
| `python xx.py` | 真实 API | mock |
| `pytest tests/` | 真实 API（CI 失败） | mock（成功） |
| `make test-llm` | 真实 API（CI 失败） | mock（成功） |
| `make ci-llm` | 真实 API | mock（Makefile 默认设 env） |
| 夜间 `make ci-real` | 真实 API | 错误用法（环境冲突） |

### 5.4 不再保留的功能

- ❌ `make run-all --tier llm` 默认走 mock —— 改成强制 `LLM_MOCK=1`
- ❌ `pytest -m "not gpu"` 不再代表"无 LLM" —— 改名 `pytest -m "not llm and not gpu"`
- ❌ `ch18/01-09` 的 `run_mock()` 函数 —— 改写为 `tests/_mocks/demo_langchain_*.py`

---

## 6. 8 Wave 详细迁移计划

| Wave | 范围 | 改动文件数 | 周期估计 | 风险 |
|------|------|----------|---------|------|
| **W1. 基建** | `shared/` + `scripts/` + `Makefile` | ~15 | 3-5 天 | 🟢 低 |
| **W2. Core tier** | 158 个 `core/*.py` | ~10 (审) | 1-2 天 | 🟢 低 |
| **W3. LLM tier** | 199 个 `llm/*.py` + 29 mock import | ~80 | 5-7 天 | 🟡 中 |
| **W4. 端侧 GPU** | `ch28/.../gpu/` 10 个 | ~10 | 2-3 天 | 🟡 中 |
| **W5. 推理引擎 GPU** | `ch25/.../gpu/` 12 个 | ~12 | 3-4 天 | 🔴 高 |
| **W6. 训练/世界模型** | `ch19/ch26/ch27/ch16` GPU | ~30 | 4-5 天 | 🔴 高 |
| **W7. 教程同步** | 29 个 .md + README | ~35 | 5-7 天 | 🟡 中 |
| **W8. CI 改造** | `.github/workflows/` | ~5 | 2-3 天 | 🟢 低 |
| **总计** | | **~200** | **25-36 天** | |

### W1 基建

**目标文件**：
- `shared/llm_client.py` — 缺 Key 抛错（~30 行改）
- `shared/provider_registry.py` — 缺 Key 抛错（~15 行改）
- `shared/gpu_guard.py` — 加 `require_nvidia_gpu / require_apple_silicon / require_ollama`（~80 行新增）
- `shared/_error_helper.py` — 新增（~40 行）
- `shared/__init__.py` — 移除 mock_llm 导出
- `scripts/llm_doctor.py` — 加 `--setup` 引导模式（~80 行新增）
- `scripts/download_models.py` — 扩到 12+ 模型清单（~60 行新增）
- `Makefile` — `test-llm / ci-llm` 加 `LLM_MOCK=1` 前缀（~10 行改）
- `tests/_mocks/` 目录新建
- `code/README.md` — 加"硬件 × 章节矩阵"段（~150 行新增）

**验收**：
- `LLM_MOCK=1 python -c "from shared.llm_client import UnifiedClient; print(UnifiedClient().is_mock)"` 走 mock
- 不设 `LLM_MOCK` 也不设 `DEEPSEEK_API_KEY` → 抛 `RuntimeError`
- `make llm-doctor --setup` 交互式引导可走完
- `make download-models --list` 列出 12+ 目标

### W2 Core tier

**目标**：审 158 个 `ch*/core/*.py`；绝大多数无 LLM 调用，仅需审计有无隐藏 mock。  
**验收**：`make ci-core` 全绿；手测 10 个代表性文件；无 `from shared.mock_llm` 残留。

### W3 LLM tier

**子任务**：
1. **下沉 mock**：`shared/mock_llm.py` → `tests/_mocks/mock_llm.py`（独立 commit）
2. **修 ch18/01-09**：9 个 `run_real()/run_mock()` 双函数 → 拆为 `01_langchain_*.py`（真实）+ `tests/_mocks/demo_*`（demo）
3. **修 ch18/13-18 + 35**：6+1 个 `if USE_REAL_API` 分支 → 默认真实
4. **修 ch22/04, ch22/11, ch29/05, ch29/12**：4 个 `if not is_real_api` → 默认真实
5. **审 ch19/02, 03**：`02_ddp_training.py` 和 `03_fsdp_training.py` 是 GPU tier 文件，本 wave **只审 + 去 mock import**，**不**做真实化（真实化留到 W6 阶段，本 wave stub 处理：调用框架改为 `accelerate launch`，但保留小模型占位 + 合成 dataset）
6. **审剩余 ~170 个 llm/*.py**：grep 全文 `MockLLM\|is_mock\|fake_llm`

**验收**：
- `LLM_MOCK=1 make ci-llm` 全绿
- 手测 5 个代表性文件
- 缺 Key 跑 → 明确报错，不静默 mock
- 有 Key 跑 → 真实 API 返回非空

### W4 端侧 GPU

**目标文件**：`ch28_edge_llm/gpu/01-10_*.py`

| 文件 | 改造 |
|------|------|
| `01-02` MLX | 真实 `mlx_lm.load/generate`；缺 Apple Silicon → RuntimeError |
| `03-04` llama.cpp | 真实 `Llama(model_path=...)`；缺权重 → RuntimeError |
| `05-06` Ollama | 删"打印代码字符串"，真实 `httpx.post` + `ollama create` |
| `07` webllm | 真实 playwright 打开 demo 页 |
| `08` webgpu/wasm | 真实 benchmark（wasmtime）|
| `09` Snapdragon NPU | 保留教学性：打印 QNN SDK 安装命令 + 示例 |
| `10` Secure Minions | 真实 TLS 协议模拟 |

**验收**：
- Apple M-series：`make test-gpu` 跑 ch28 真实测试
- 缺硬件：`python xx.py` 抛 RuntimeError

### W5 推理引擎 GPU

**目标文件**：`ch25_inference_engines/gpu/01-12_*.py`

| 文件 | 改造 |
|------|------|
| `01-04, 06, 07` vLLM 内部类 | 用 `vllm.block_manager / prefix / speculative / expert_parallel` 真实类 |
| `05` KV cache 计算 | 纯计算，仅审 |
| `08` TensorRT-LLM | **改名** `08_tensorrt_llm_build.py`；真实 `trtllm-build` |
| `09` FP4 量化 | 真实 `bitsandbytes` / `transformers` |
| `10` vLLM Async Engine | **改名** `10_vllm_async_engine.py`；真实 `vllm.AsyncLLMEngine` |
| `11` SLO 监控 | 真实 `prometheus_client` |
| `12` 引擎选择 | 决策函数（纯逻辑）|

**验收**：
- NVIDIA 24GB+：`vllm.AsyncLLMEngine.from_engine_args(Qwen2.5-7B)` 真实启动
- 缺 GPU / 缺权重 → RuntimeError 明确

### W6 训练/世界模型

**目标**：
- `ch19/02,03` 真实 `accelerate launch` 跑 Qwen2.5-0.5B
- `ch26/01-10` 真实加载 Cosmos/Pi0 跑 forward
- `ch16/01-sft,02-lora` 真实 `trl.SFTTrainer` 跑 10 step

**验收**：NVIDIA 24GB+ `accelerate launch --num_processes 2` 跑通

### W7 教程同步

**重写章节**（实质性更新）：
- `00_目录索引.md` — 加"硬件 × 章节矩阵"表
- `18_LLM工程框架实战.md` — 大量改写；删 mock demo 段
- `25_推理引擎与高性能服务.md` — 加 vLLM 真实启动 + 硬件需求
- `26_世界模型与具身AI.md` — 加 Cosmos/Pi0 真实加载段
- `27_推理模型与Test-Time_Compute.md` — 加 R1/Distill 真实运行段
- `28_端侧与边缘LLM.md` — 加 Ollama 真实调用 + MLX 真实段
- `17_大模型评估体系.md` — 改"如何评估"段

**小改章节**：Ch13/14/15/16/20/22/29 替换"运行例子"段

**README**：
- `code/README.md` — 加"硬件 × 章节矩阵"表 + "环境配置"段
- `README.md` — 加"配套代码使用指南"段

**验收**：`make verify-xrefs` 全绿；用户按教程能复现

### W8 CI 改造

**目标文件**：
- `.github/workflows/pr-check.yml` — PR 冒烟（mock 路径）
- `.github/workflows/integration-test.yml` — 夜间真实 API
- `.github/workflows/gpu-smoke.yml` — 夜间 GPU 真实（self-hosted runner）
- `scripts/verify_all.py` — 加 LLM_MOCK 检查

**新增 secrets**：`DEEPSEEK_API_KEY`、可选 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`

**自托管 runner**：1 台 24GB NVIDIA GPU 机器，仅 `gpu-smoke.yml` 使用

**验收**：PR check ≤ 3 分钟；夜间 integration test 跑 10 个真实 API 文件；self-hosted 跑 5 个 GPU 例子

---

## 7. 风险与回滚

| 风险 | 缓解 |
|------|------|
| **W3 漏改 1 个文件** 导致主流程仍走 mock | `grep -rln "is_mock\|fake_llm\|MockLLM()" --include="*.py" code/` 在 W3 末尾 + W4-W6 每个 wave 末尾跑 |
| **W5/W6 真实 GPU 例子在 CI 上 OOM** | `require_nvidia_gpu(min_vram_gb=24)` 前置检查；夜间 GPU workflow 仅 self-hosted runner 跑 |
| **真实 API 响应不稳定导致 CI 抖动** | PR 检查 100% 走 mock；真实 API 仅夜间跑 |
| **教程与代码反向链接断链** | `make verify-xrefs` 必须在 W7 每个 commit 后跑 |
| **下载脚本在海外用户环境失败** | `download_models.py` 已支持 ModelScope / hf-mirror / HF 直连三源 |
| **W4/W5/W6 周期超 25-36 天估计** | 每波独立 PR 可独立合并；超期可拆 W6 为 W6a/W6b |

---

## 8. 成功标准

满足以下全部条件，视为本次重构完成：

- [ ] `code/` 主流程中 `grep "is_mock\|fake_llm\|MockLLM()" code/` 输出 0 行
- [ ] `python xx.py` 在 4 类用户环境下全部跑通：
  - [ ] 有 `DEEPSEEK_API_KEY` 的国内用户（默认场景）
  - [ ] Apple M-series + Ollama 已启动
  - [ ] NVIDIA 24GB+ + vLLM 权重已下载
  - [ ] 无 Key + `LLM_MOCK=1`（CI / 离线）
- [ ] `make ci-core` / `make ci-llm` (mock) / 夜间 integration test 全绿
- [ ] 29 个 .md 教程反向链接全通
- [ ] 仓库 README "硬件 × 章节矩阵"表完整

---

## 9. 不在本设计范围内

- ❌ 改教程非代码章节（如面试题答案、概念解释）
- ❌ 添加新教程章节（仅维护现有 29 章）
- ❌ 替换 mock_llm 的实现（直接下沉到 tests/_mocks/，不再扩展）
- ❌ 引入新的 LLM 厂商（仅在 6 家中选）
- ❌ 改测试策略为端到端（仍以单文件 smoke test 为主）
