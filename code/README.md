# Code Companion — 可运行代码伴侣

> 教程 [`../README.md`](../README.md) 中 29 章 / 384 段代码的可执行版

本目录是 Obsidian 教程库（29 章节 ~2,200 KB）的**配套可运行代码库**。每段教程中的 Python 代码被整理为**端到端可运行的 `.py` 文件**，带 frontmatter 注释、跨章节交叉引用、smoke tests。

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
python -m venv .venv && source .venv/Scripts/activate  # Windows
make install-core
python ch12_transformer_architecture/core/01_scaled_dot_product_attention.py
```

### 跑真实 LLM 例子 (推荐)

```bash
cd code/
export DEEPSEEK_API_KEY=sk-xxx          # 推荐 DeepSeek (国内快, OpenAI 协议, 注册送 ¥10)
make install-llm                         # 5 分钟
make download-models-default             # 1.7GB: bge + reranker + Qwen2.5-0.5B
LLM_MOCK=1 make test-llm                 # mock 冒烟
python ch15_agent/llm/01_react_basic.py  # 真实跑
```

→ 详细见 [QUICKSTART.md](./QUICKSTART.md)

## 目录结构

```
code/
├── README.md (this)
├── QUICKSTART.md
├── Makefile
├── pyproject.toml
├── requirements-{core,llm,gpu}.txt
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
make test          # core/ 全部 (~110 个例子)
make test-llm      # llm/ 全部 (~280 个例子, 用 mock)
make test-gpu      # gpu/ 全部 (~80 个例子, 需 CUDA)
```

## 🖥️ 硬件 × 章节矩阵

按你的硬件选章节（其他章节也能跑，但需要 API Key）：

| 硬件 | 可跑章节 | 必装命令 |
|------|---------|---------|
| **任意笔记本** (无 GPU) | Ch1-24 全部 core + llm tier | `pip install -r requirements-llm.txt` |
| **+ DeepSeek API Key** | 同上 + 真实 LLM 输出 | `make llm-doctor-setup` |
| **Apple M-series** (≥8GB) | Ch28 端侧 (MLX / Ollama) | `brew install ollama && ollama pull llama3.2:3b` |
| **NVIDIA GPU 8GB** | + Ch19 (DDP 0.5B) + Ch16 (LoRA 0.5B) | `make install-gpu` |
| **NVIDIA GPU 24GB+** | + Ch25 (vLLM 7B) + Ch27 (R1-Distill 1.5B) | `make download-models-llm` |
| **NVIDIA GPU 80GB (A100/H100)** | + Ch26 (Cosmos 7B + Pi0) | `make download-models-gpu` |

### 下载脚本速查

```bash
make download-models-list          # 列出所有可选模型
make download-models-default       # 1.7GB: embedding + reranker + 0.5B LLM
make download-models-llm           # +30GB: 7B/8B LLM (vLLM 用)
make download-models-gpu           # +25GB: 世界模型 / VLA / 推理
make download-models-edge          # +7GB: MLX / GGUF 端侧
```

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
| Ch26 世界模型 | NVIDIA 80GB | cosmos7b + pi0 | 无 (本地) |
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

# 3. 下载模型权重
make download-models-default  # 1.7GB, 必须

# 4. 验证
LLM_MOCK=1 make test-llm     # mock 测试
DEEPSEEK_API_KEY=sk-xxx python ch15_agent/llm/01_react_basic.py  # 真实跑
```

## ✅ W1-W6 真实化状态 (35 commits)

450 个 .py 例子 W1-W6 完成真实化重构, 默认走真实 API/模型, mock 仅作 fallback.

| 章节 | 真实跑内容 | 缺什么时降级 |
|------|-----------|-------------|
| Ch1-11 (Python/ML 基础) | 纯 Python / numpy / sklearn | 无依赖 |
| Ch12-18 (LLM 基础) | DeepSeek (OpenAI 协议) | `LLM_MOCK=1` 走 mock |
| Ch14 (RAG) | 本地 bge + ChromaDB + DeepSeek | mock embedder |
| Ch17 (评估) | RAGAS + DeepSeek LLM judge | mock judge |
| Ch19 (DDP) | Qwen2.5-0.5B + DDP 多卡 | 24GB×2 卡 |
| Ch20 (LLMOps) | LangFuse v3 + DeepSeek | mock trace |
| Ch25 (推理引擎) | vLLM + Qwen2.5-7B 量化 | 24GB GPU |
| Ch26 (世界模型) | Cosmos-7B config + flow matching | 80GB GPU |
| Ch27 (推理模型) | DeepSeek-R1 API + R1-Distill-1.5B | 24GB GPU |
| Ch28 (端侧) | Ollama (本机) + MLX | Apple Silicon / Ollama |

**关键原则**:
- `export DEEPSEEK_API_KEY=sk-xxx` 是默认推荐配置 (国内快, OpenAI 协议, ¥10 体验金)
- 无 Key 时 `LLM_MOCK=1` 走 mock, 不报错
- 缺权重时友好 `RuntimeError` + `make llm-doctor-setup` 提示
