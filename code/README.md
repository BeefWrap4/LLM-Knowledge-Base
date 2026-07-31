# Code Companion — 可运行代码伴侣

> 40 章教程中 29 章的代码伴侣：433 个示例（158 core + 199 llm + 76 gpu）

本目录是 40 章 Obsidian 教程中 29 章的**配套可运行代码库**。示例带 metadata、
章节引用和 smoke tests；执行状态以本页的验收命令为准。

## 三大层级（tier）

| Tier | 安装时间 | 大小 | 适用章节 | 典型例子 |
|------|---------|------|---------|---------|
| **core** | 30 秒 | 50MB | Ch1-11 (Python/ML 基础) | 装饰器、上下文管理器、Attention |
| **llm**  | +5 分钟 | 500MB | Ch12-24 (LLM 工程) | LangChain、RAG、Agent |
| **gpu**  | +30 分钟 | 8GB+ | Ch25-29 (推理引擎、2026 主题) | vLLM、SGLang、MLX、LeRobot |

**推荐路径**: core → llm → gpu。CPU 笔记本用户 80% 例子可跑。

## 快速上手

```bash
cd code/
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# Bash: source .venv/bin/activate
make install-core
python ch12_transformer_architecture/core/01_scaled_dot_product_attention.py
```

### 默认离线与条件性真实 LLM

```bash
cd code/
make install-llm
LLM_MOCK=1 make test-llm                 # 默认离线，不读 Key、不联网

# 仅在明确接受数据出域与计费后运行
export DEEPSEEK_API_KEY=your-key
bash scripts/run_real_demos.sh --confirm-real quick deepseek
```

下载模型、真实 API、GPU 和外部服务分别验收；不要把 `[SKIP]` 或离线 mock 计为真实通过。

→ 详细见 [QUICKSTART.md](./QUICKSTART.md)

## 目录结构

```
code/
├── README.md (this)
├── QUICKSTART.md
├── Makefile
├── pyproject.toml
├── requirements-{core,llm,gpu}.txt       # 本地分层依赖（带主版本上界）
├── requirements-{core,llm,gpu}.ci.lock   # Python 3.12/Linux CI 锁文件
├── shared/                 # 跨章节工具 (gpu_guard, mock_llm, env)
├── ch01_*/ ... ch29_*/     # 一章一目录，含 core/llm/gpu/ 三个子目录
└── tests/                  # pytest smoke tests
```

## 与教程的关系

每个 `.py` 文件头部的注释含：

```python
# ---
# chapter: 12
# topic: Scaled Dot-Product Attention
# section: 12.2.5
# ---
# See: ../tutorial/Ch12_Transformer与大模型原理.md §12.2.5
```

用相对路径 `../tutorial/` 引用教程章节。**教程文件不被修改**——所有引用是单向的。

## 验证

```bash
make test          # pytest（默认排除 GPU）
make ci-core       # core/ 158 个示例
LLM_MOCK=1 make ci-llm  # llm/ 199 个示例，离线 mock
make test-gpu      # pytest（包含 gpu marker），需 NVIDIA GPU
python scripts/run_all_examples.py --tier gpu  # 76 个脚本的离线契约；条件项记为 SKIP
python scripts/run_all_examples.py --tier gpu --chapter ch16 --real-gpu  # 真实运行选定章节
```

`gpu/` 同时包含 NVIDIA、Apple Silicon、Ollama 和浏览器示例，单台机器不应宣称真实跑通
全部 76 个。默认 runner 传入 `--mock` 并分别统计 Passed/Skipped/Failed；
`--real-gpu` 只关闭默认 mock，仍不会自动授权下载、端口监听、引擎编译、浏览器或
云部署；这些路径需要各脚本声明的独立环境开关。只应在隔离环境中按 `--chapter`
串行运行兼容子集。

## 🖥️ 硬件 × 章节矩阵

按你的硬件选条件子集；显存档位不构成跑通承诺，还要核对模型、许可证、驱动/CUDA、
上下文/并发和引擎兼容矩阵：

| 硬件 | 可跑章节 | 必装命令 |
|------|---------|---------|
| **任意笔记本** (无 GPU) | 158 个 core + 支持 mock 的 llm 示例 | `pip install -r requirements-llm.txt` |
| **+ 供应商 API Key** | 显式 `LLM_MOCK=0` 后运行对应真实 LLM 子集 | `make llm-doctor-setup` |
| **Apple M-series** | Ch28 的 MLX/Ollama 兼容子集 | 按官方文档安装并选择适配统一内存的模型 |
| **NVIDIA GPU** | Ch16/19/21/25/26 的兼容子集 | `make install-gpu`；先做容量预算 |
| **多 GPU / 高速互联** | DDP/FSDP、TP/EP、PD 分离子集 | 核对 NCCL/NIXL、拓扑和启动方式 |

### 下载脚本速查

```bash
make download-models-list          # 列出所有可选模型
make download-models-default       # 下载前查看清单、大小、许可证与剩余磁盘
make download-models-llm           # LLM 条件子集
make download-models-gpu           # 世界模型 / VLA / 推理条件子集
make download-models-edge          # MLX / GGUF 端侧条件子集
```

## 🐳 Docker vLLM 集成 (Windows escape hatch)

`vllm._C` 编译扩展在 Windows 上缺失, 7 个 vLLM 例子默认友好抛错. **Docker escape hatch** 让 Windows 用户也能真跑:

### 1. 启动 vLLM server (后台容器)

```bash
# 默认模型也需要先确认本地已下载/可访问；启动时间依机器与模型而异
make vllm-server-start

# 自定义模型 (e.g. Qwen2.5-7B 需先下: make download-models-llm)
MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8001 make vllm-server-start
```

容器自动:
- 用 `vllm/vllm-openai:latest` 镜像
- GPU passthrough (`--gpus all`)
- 挂载 `code/models/` 到 `/root/.cache/huggingface/` (免重下)
- 轮询健康端点并以 `/v1/models` 成功响应作为 ready 条件

### 2. 用 OpenAI 客户端连 server (绕开 vllm._C)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="/root/.cache/huggingface/Qwen2.5-0.5B-Instruct",  # 容器内路径
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=32,
)
print(resp.choices[0].message.content)
```

### 3. 验证 / 停止

```bash
make vllm-server-status    # 检查 server 是否运行
make vllm-server-stop     # 停止容器
```

### 4. 7 个 vLLM 例子的兼容性

| 文件 | vLLM._C 缺失 | Docker server |
|------|--------------|---------------|
| `ch25/01-04` (paged/continuous/radix/PD) | 友好抛错 | 可改用 OpenAI 客户端 (需重构) |
| `ch25/06, 07` (speculative, MoE) | 友好抛错 | 同上 |
| `ch25/10` (vLLM async engine) | 友好抛错 | 同上 |
| `ch18/13-18, 35` (LlamaIndex) | 友好抛错 | 同上 |

**当前状态**: Docker 启动脚本是 escape hatch, 例子本身未重构. 用户可自行写 OpenAI 客户端代码连 server.

### 5. Linux/WSL2/Docker 镜像推荐 (生产用)

| 环境 | 推荐命令 |
|------|---------|
| Linux (有 GPU) | `pip install vllm` (官方支持, 编译扩展正常) |
| WSL2 (有 GPU) | `pip install vllm` (Windows Subsystem 内部 Linux) |
| Docker | `vllm/vllm-openai:0.21.0` (本脚本) |
| 云端 (RunPod/Vast) | 用 vLLM 官方模板 + `pip install vllm` |

---

注: 这是 Windows 用户的临时 escape hatch. **生产推荐** 仍在 Linux 上 `pip install vllm`.

### 各章节最低硬件需求

| 章节 | 最低硬件 | 模型权重 | 推荐 API |
|------|---------|---------|---------|
| Ch1-11 | 任意 | 无 | 无 |
| Ch12 Transformer | 任意 | 无 | 无 |
| Ch13 Prompt | 任意 | 无 | DeepSeek / OpenAI |
| Ch14 RAG | 任意 | bge-small-zh | DeepSeek |
| Ch15 Agent | 任意 | 无 | DeepSeek / Anthropic |
| Ch16 微调 | NVIDIA 8GB | qwen0.5b | DeepSeek |
| Ch17 评估 | 任意 | bge-small-zh + reranker | DeepSeek |
| Ch18 LLM 框架 | 任意 | 无 | DeepSeek / OpenAI |
| Ch19 分布式 | NVIDIA 24GB × 2 卡 (DDP) | qwen0.5b × 2 | DeepSeek |
| Ch20 LLMOps | 任意 | 无 | DeepSeek |
| Ch21-23 | 任意 | bge (部分) | DeepSeek |
| Ch24 云原生 | 任意 | 无 | DeepSeek |
| Ch25 推理引擎 | NVIDIA 24GB | qwen7b | 无 (本地) |
| Ch26 世界模型 | 按当前模型卡与目标任务核算 | Cosmos 3 / Pi0 需显式确认后另行下载 | 无（默认离线跳过） |
| Ch27 推理模型 | NVIDIA 24GB (本地 r1-distill) 或 DeepSeek-R1 API (云端) | r1-distill-1.5b | DeepSeek-R1 API |
| Ch28 端侧 | Apple Silicon | mlx-qwen7b / gguf-llama3b | Ollama (本地) |
| Ch29 Context Eng | 任意 | 无 | DeepSeek |

### 完整环境配置

```bash
# 1. 安装依赖
make install-llm        # 5 分钟, 无需 GPU

# 2. 配置 API Key (推荐 DeepSeek)
make llm-doctor-setup   # 交互式
# 或手动: export DEEPSEEK_API_KEY=sk-xxx
# 真实 Key 探针（明确 provider 且可能计费）
make llm-doctor-check PROVIDER=deepseek

# 3. 下载模型权重
make download-models-default  # 默认 required 集合；实际体积以当前 revision 为准

# 4. 验证
LLM_MOCK=1 make test-llm     # mock 测试
LLM_MOCK=0 LLM_PROVIDER=deepseek DEEPSEEK_API_KEY=sk-xxx \
  python ch15_agent/llm/02_react_agent_from_scratch.py  # 显式真实跑
```

## 代码伴侣状态

当前共 433 个 `.py` 示例。批量 `llm` 验收默认使用 `LLM_MOCK=1`；真实 API、
本地模型和 GPU 路径必须显式配置，并与离线 CI 分开验证。

| 章节 | 真实跑内容 | 缺什么时降级 |
|------|-----------|-------------|
| Ch1-11 (Python/ML 基础) | 纯 Python / numpy / sklearn | 无依赖 |
| Ch12-18 (LLM 基础) | DeepSeek (OpenAI 协议) | `LLM_MOCK=1` 走 mock |
| Ch14 (RAG) | 本地 bge + ChromaDB + DeepSeek | mock embedder |
| Ch17 (评估) | RAGAS + DeepSeek LLM judge | mock judge |
| Ch19 (DDP) | Qwen2.5-0.5B + DDP 多卡 | 24GB×2 卡 |
| Ch20 (LLMOps) | LangFuse v3 + DeepSeek | mock trace |
| Ch25 (推理引擎) | 默认离线契约；条件路径接入 vLLM/量化 | 容量按模型、上下文、并发与引擎版本估算 |
| Ch26 (世界模型) | Cosmos-7B config + flow matching | 80GB GPU |
| Ch27 (推理模型) | DeepSeek-R1 API + R1-Distill-1.5B | 24GB GPU |
| Ch28 (端侧) | Ollama (本机) + MLX | Apple Silicon / Ollama |

**关键原则**:
- API Key 只通过环境变量或本地 `.env` 提供，禁止提交到仓库
- 无 Key 时 `LLM_MOCK=1` 走 mock, 不报错
- 缺权重时友好 `RuntimeError` + `make llm-doctor-setup` 提示
