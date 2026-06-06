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
