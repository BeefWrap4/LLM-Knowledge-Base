# Python 到大模型应用 — 面试教程 2026 版

> **面向 2026 年大模型算法/工程岗位的 Python 全栈面试准备库**
>
> 29 章节 · 280+ 面试题 · 2026 年热点全覆盖 · 健康评分 95/100

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

### 在 Obsidian 中阅读（推荐）

1. 下载 [Obsidian](https://obsidian.md/)
2. 打开本仓库目录作为 Vault
3. 从 `00_目录索引.md` 开始浏览
4. 利用 Obsidian 的图谱视图（Graph View）查看知识关联

### 在其他 Markdown 阅读器中

直接打开任何 `.md` 文件即可阅读。Wiki 链接 `[[章节名]]` 在大部分阅读器中无法跳转，请使用 Obsidian 获得完整体验。

### 搜索特定考点

所有 Wiki 链接使用 `[[NN_TopicName]]` 格式，搜索 `[[16_` 即可找到所有指向第 16 章的反向链接。

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
| 总文件数 | 32 (29 章正文 + TOC + 健康报告 + README) |
| 总大小 | ~2,200 KB |
| 总面试题 | 300+ 道 |
| 总 Mermaid 图 | 200+ 个 |
| 总代码示例 | 500+ 段 |
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

教程中 384 段 Python 代码已整理为 **端到端可运行的 .py 文件**，位于 [`code/`](code/) 目录：

```bash
cd code/
python -m venv .venv && source .venv/Scripts/activate
make install-core                                          # 30 秒
python ch12_transformer_architecture/core/01_scaled_dot_product_attention.py
```

支持三层依赖：
- **core** — 任何电脑，30 秒安装
- **llm** — API 调用，5 分钟安装
- **gpu** — 需 NVIDIA GPU，30 分钟安装

每个示例反向引用教程章节，5 分钟从 clone 到第一次运行。详见 [code/QUICKSTART.md](code/QUICKSTART.md)。

## 📝 维护与贡献

- 每月一次全库健康审计（见 `99_库健康检查报告.md`）
- 每新增 5 章做一次内容质量评估
- 章节命名严格遵循 `NN_TopicName.md` 两位数字前缀
- 新章节必须包含：YAML frontmatter / Mermaid 图 / 面试题 / 速查表 / 交叉引用

## 📜 许可

本教程仅供学习用途。涉及的开源框架（vLLM, LangChain, DeepSpeed 等）遵循各自原项目的许可协议。

## 🆕 更新日志

- **2026-06-06** — 新增 Ch25-29 五个 2026 年新主题章节；库扩展至 29 章，健康评分 95/100
- **2026-06-02** — Mindmap 统一改为文本树；Wiki 链接修复；交叉引用网络完善
- **2026-06-01** — 初始版本：24 章核心内容

---

*由 Claude Code 辅助维护。最新一次更新见 git log。*
