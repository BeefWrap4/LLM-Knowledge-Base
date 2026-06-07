# Python 到大模型应用 — 面试教程 2026 版

> **面向 2026 年大模型算法/工程岗位的 Python 全栈面试准备库**
>
> 29 章节 · 280+ 面试题 · 439 个可运行代码示例 · 2026 主题全覆盖 · 健康评分 95/100

[![Verify](https://github.com/BeefWrap4/LLM-Knowledge-Base/actions/workflows/verify.yml/badge.svg)](https://github.com/BeefWrap4/LLM-Knowledge-Base/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-green.svg)](https://www.python.org/)
[![code passing](https://img.shields.io/badge/code-357%2F357-brightgreen.svg)]()
[![chapters](https://img.shields.io/badge/chapters-29%2F29-blue.svg)]()
[![vendors](https://img.shields.io/badge/LLM%20vendors-7-orange.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)
[![CN mirror](https://img.shields.io/badge/CN%20mirror-清华%2BModelScope-red.svg)]()

本仓库是一份面向 **2026 年大模型算法/应用/部署/安全** 岗位的面试准备教程，**完全基于 Obsidian 知识库结构组织**。从 Python 基础到 LLM 工程实践，覆盖大模型开发的完整技术栈。

## ✨ 核心特色

- 🆕 **2026 主题全覆盖** — Reasoning Models, Test-Time Compute, MCP/A2A, Inference Engines, World Models, Context Engineering
- 📊 **300+ 真实面试题** — 每章含 5-16 道详细题解，含参考答\n- 🔗 **完整交叉引用网络** — Wiki 链接形式，与 Obsidian 原生集成
- 📋 **100% 风格统一** — 每章含 YAML frontmatter / Mermaid 图 / 速查表 / 面试题 / 交叉引用
- 🛠️ **可执行代码示例** — Python/Shell/YAML/Dockerfile 完整可运行

## 📚 知识体系

```
🐍 Python 编程基础 (6 章)
├── 第1章  Python 编程基础
├── 第2章  可变性与拷贝
├── 第3章  面向对象编程
├── 第4章  高级特性与函数式
├── 第5章  并发编程
└── 第6章  内存管理与 GC

📊 数据科学与算法 (2 章)
├── 第7章  数据结构与算法
└── 第8章  数据科学核心库

🌐 Web 开发 (1 章)
└── 第9章  Web 开发与 FastAPI

🧠 机器学习与深度学习 (2 章)
├── 第10章 机器学习基础
└── 第11章 深度学习与 PyTorch

🤖 大模型核心技术 (5 章)
├── 第12章 Transformer 与大模型原理
├── 第13章 Prompt Engineering
├── 第14章 RAG 检索增强生成
├── 第15章 Agent 智能体开发
└── 第16章 模型微调与推理优化

🔧 大模型工程实践 (13 章)
├── 第17章 大模型评估体系
├── 第18章 LLM 工程框架实战
├── 第19章 分布式训练系统
├── 第20章 LLMOps 与可观测性
├── 第21章 多模态大模型
├── 第22章 大模型数据工程
├── 第23章 AI 安全与伦理
├── 第24章 云原生部署与工程化
├── 🆕 第25章 推理引擎与高性能服务
├── 🆕 第26章 世界模型与具身 AI
├── 🆕 第27章 推理模型与 Test-Time Compute
├── 🆕 第28章 端侧与边缘 LLM
└── 🆕 第29章 Context Engineering
```

## 🎯 按岗位选择学习路径

### 大模型算法工程师
**必读**: 1-7, 10-12, 16, 19, 22
**选读**: 13, 14, 17, 21, 27

### 大模型应用开发工程师
**必读**: 1-6, 9, 13-15, 18, 20, 29
**选读**: 7, 12, 16, 17, 25, 28

### 大模型推理部署工程师 (ML Platform/SRE)
**必读**: 5, 9, 16, 19, 20, 24, 25
**选读**: 12, 14, 15, 28

### 大模型评估工程师 (Evals/Reliability)
**必读**: 12, 14, 15, 17, 20, 27
**选读**: 16, 22, 23, 25

### Agent 工程师 (2026 热门)
**必读**: 12, 14, 15, 18, 25, 27, 29
**选读**: 17, 20, 22, 23

### 具身智能 / 机器人工程师
**必读**: 11, 21, 26
**选读**: 12, 15, 19

## 🚀 快速开始

### 5 分钟跑通第一个代码例子 (推荐)

```bash
git clone https://github.com/BeefWrap4/LLM-Knowledge-Base.git
cd LLM-Knowledge-Base/code
python -m venv .venv && source .venv/Scripts/activate
make install-core                    # 30 秒, CPU 即可
python ch12_transformer_architecture/core/01_scaled_dot_product_attention.py
# 输出: OK
```

跑通后, 任意 `ch*/llm/*.py` 或 `ch*/gpu/*.py` 都可执行, 三层依赖自动决定.

### 三层依赖策略 (按需升级)

| Tier | 安装命令 | 适用例子 | 备注 |
|------|---------|---------|------|
| **core (158)** | `make install-core` | Ch01-11 | 任何电脑, 30s |
| **llm (199)** | `make install-llm` | Ch12-24 | mock 模式或 API key, 5min |
| **gpu (76)** | `make install-gpu` | Ch25-29 | NVIDIA GPU + 重型依赖, 30min |

### 在 Obsidian 中阅读（推荐）

1. 下载 [Obsidian](https://obsidian.md/)
2. 打开本仓库目录作为 Vault
3. 从 `00_目录索引.md` 开始浏览
4. 利用 Obsidian 的图谱视图（Graph View）查看知识关联

### 在其他 Markdown 阅读器中

直接打开任何 `.md` 文件即可阅读。Wiki 链接 `[[章节名]]` 在大部分阅读器中无法跳转，请使用 Obsidian 获得完整体验。

### 搜索特定考点

所有 Wiki 链接使用 `[[NN_TopicName]]` 格式，搜索 `[[16_` 即可找到所有指向第 16 章的反向链接。

### 一键验证 5 项检查

```bash
cd code/
make verify                # <30s, 任何 commit 前跑一次
make ci                    # ~10min, push 前跑 (镜像 GitHub Actions)
```

## 📊 2026 主题速查

| 主题 | 所在章节 | 面试频率 |
|------|----------|---------|
| MCP / A2A 协议 | Ch15.4 / Ch15.9 | ⭐⭐⭐⭐⭐ |
| Reasoning Models (o3/R1) | Ch12 / Ch27 | ⭐⭐⭐⭐⭐ |
| Test-Time Compute | Ch12.8 / Ch16.8 / Ch27 | ⭐⭐⭐⭐⭐ |
| PagedAttention / KV Cache | Ch16 / Ch25 | ⭐⭐⭐⭐⭐ |
| vLLM / SGLang | Ch25 | ⭐⭐⭐⭐⭐ |
| LoRA / QLoRA / GRPO | Ch12 / Ch16 | ⭐⭐⭐⭐⭐ |
| RAG (ColPali, GraphRAG) | Ch14 | ⭐⭐⭐⭐⭐ |
| Agent Skills (SKILL.md) | Ch15.8 | ⭐⭐⭐⭐ |
| Pydantic AI / Strands | Ch18.8 | ⭐⭐⭐⭐ |
| World Models / VLA | Ch21 / Ch26 | ⭐⭐⭐⭐ |
| Apple MLX / 端侧 LLM | Ch28 | ⭐⭐⭐ |
| Context Engineering | Ch29 | ⭐⭐⭐⭐ |

## 🛠️ 工具栈

- **格式**: Obsidian Flavored Markdown
- **图表**: Mermaid 10+ 类型 (flowchart, sequenceDiagram, timeline 等)
- **数学**: LaTeX `$` / `$$` 块
- **代码**: Python 3.13+ 标准库 + 流行框架示例
- **引用**: `[[Wiki Links]]` 形式，无 markdown 链接

## 📈 库统计

| 指标 | 数值 |
|------|------|
| 教程总文件数 | 33 (29 章正文 + TOC + 健康报告 + README + CLAUDE.md) |
| 教程总大小 | ~2,200 KB |
| 总面试题 | 300+ 道 |
| 总 Mermaid 图 | 200+ 个 |
| 总代码示例 | 500+ 段 |
| **code/ 伴侣 .py 文件** | **439 (158 core/ + 199 llm/ + 76 gpu/)** |
| **code/ 章节 README** | **29/29 (100%)** |
| 健康评分 | 95/100 |
| 风格一致性 | 100% |

## 🔧 仓库结构

```
.
├── 00_目录索引.md                    # 导航中枢，按板块/岗位多维组织
├── 01-24_*.md                        # 24 个核心章节
├── 25-29_*.md                        # 5 个 2026 新增章节
├── 99_库健康检查报告.md              # 维护记录与质量报告
├── CLAUDE.md                         # Claude Code 工作指南
└── README.md                         # 本文件
```

## 🛠️ 可运行代码伴侣

教程中所有 Python 代码已整理为 **439 个端到端可运行的 .py 文件**, 位于 [`code/`](code/) 目录:

```bash
cd code/
python -m venv .venv && source .venv/Scripts/activate
make install-core                                          # 30 秒
python ch12_transformer_architecture/core/01_scaled_dot_product_attention.py
```

**三层依赖策略** (覆盖 100% 用户场景):
- **core (158 文件)** — 任何电脑, 30 秒安装
- **llm (199 文件)** — API 调用 + mock 模式, 5 分钟安装
- **gpu (76 文件)** — 需 NVIDIA GPU/Apple MLX, 30 分钟安装

### 接入真实 LLM (Wave 14-A)

```bash
cd code/
cp .env.example .env        # 编辑填入 DEEPSEEK_API_KEY 或 KIMI_API_KEY
python scripts/llm_doctor.py  # 验证 Key
```

支持 **DeepSeek / Kimi / SiliconFlow / OpenAI / Anthropic** 多厂商 (统一 OpenAI 兼容协议).
详见 [`code/docs/API_KEYS.md`](code/docs/API_KEYS.md).

### 下载教程所需模型 (Wave 14-B)

```bash
cd code/
make download-models       # 国内源 (ModelScope), 默认下 bge-small + bge-reranker
```

详见 [`code/models/README.md`](code/models/README.md) 和 [`code/docs/MODELS.md`](code/docs/MODELS.md).

### Docker 化部署 (Wave 14-C)

```bash
make -C code docker-build    # 构建镜像 (国内源加速)
make -C code docker-llm      # 启动 app + Redis
make -C code docker-bash     # 进容器
```

3 个 profile: `core` / `llm` (+Redis) / `gpu` (+pgvector). 详见 [`code/docs/DEPLOY.md`](code/docs/DEPLOY.md).

### 拉取预构建 Docker 镜像 (Wave 18)

```bash
# GitHub Container Registry (自动 build, push by .github/workflows/docker-build.yml)
docker pull ghcr.io/beefwrap4/llm-knowledge-base:latest

# 跑
docker run --rm -it \
  -e DEEPSEEK_API_KEY="sk-xxx" \
  -e KIMI_API_KEY="sk-xxx" \
  -e MINIMAX_API_KEY="sk-cp-xxx" \
  -v ${PWD}/code/models:/app/code/models \
  ghcr.io/beefwrap4/llm-knowledge-base:latest bash

# 容器内:
cd /app/code && make ci-quick   # 验证 5 项检查 + 真实 LLM 调用
```

### CI 配置真实 LLM API Key (Wave 18)

仓库维护者: 在 GitHub → Settings → Secrets and variables → Actions 添加 (任选 1 个):

| Secret 名 | 厂商 |
|----------|------|
| `DEEPSEEK_API_KEY` | DeepSeek |
| `KIMI_API_KEY` | Moonshot Kimi |
| `SILICONFLOW_API_KEY` | 硅基流动 |
| `MINIMAX_API_KEY` | MiniMax (Codin Plan) |
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |

配置后, [ci-llm-doctor workflow](.github/workflows/ci-llm-doctor.yml) 会每周一 14:00 自动跑真实调用, 也可手动触发.

## 📝 维护与贡献

- 每月一次全库健康审计（见 `99_库健康检查报告.md`）
- 每新增 5 章做一次内容质量评估
- 章节命名严格遵循 `NN_TopicName.md` 两位数字前缀
- 新章节必须包含：YAML frontmatter / Mermaid 图 / 面试题 / 速查表 / 交叉引用

## 📜 许可

本教程仅供学习用途。涉及的开源框架（vLLM, LangChain, DeepSpeed 等）遵循各自原项目的许可协议。

## 🆕 更新日志

- **2026-06-07** — code/ 伴侣 Phase 0-3 完成: 439 个 .py 文件 + 29 个章节 README + 3 层 tier 依赖 + tutorial/ junction
- **2026-06-06** — 新增 Ch25-29 五个 2026 年新主题章节；库扩展至 29 章，健康评分 95/100
- **2026-06-02** — Mindmap 统一改为文本树；Wiki 链接修复；交叉引用网络完善
- **2026-06-01** — 初始版本：24 章核心内容

---

*由 Claude Code 辅助维护。最新一次更新见 git log。*
