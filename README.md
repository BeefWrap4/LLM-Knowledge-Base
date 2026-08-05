# Python 到大模型应用面试教程（2026 版）

> 54 章节 · 8 大部分 · 433 个可运行代码示例 · 事实基线：2026-07-31 · 结构审校：2026-08-05

[![chapters](https://img.shields.io/badge/chapters-54-blue.svg)](00_目录索引.md)
[![examples](https://img.shields.io/badge/examples-433-success.svg)](code/README.md)
[![python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](code/README.md)

这套教程把 Python、模型原理、RAG、Agent、训练、推理服务和岗位面试组织成一条可验证的学习链。每章都有导航、导入段、连续编号正文、自测、代码验收、面试表达、速查表和一手来源入口。

## 从哪里开始

- 第一次阅读：打开 [[00_目录索引]]，按 16 周主线推进。
- 已有 Python 基础：从 [[11_机器学习基础]] 或 [[13_Tokenizer与词表工程]] 开始。
- 只做 LLM 应用：重点阅读 Ch17–Ch28、Ch36–Ch45 和 Ch54。
- 只做训练 / Infra：重点阅读 Ch11–Ch16、Ch29–Ch46、Ch51。

## 54 章结构

| 分部 | 范围 | 学习产出 |
|---|---:|---|
| 第一部分 Python 与后端工程基础 | Ch01–Ch10 | 建立能运行、能调试、能服务化的 Python 基座 |
| 第二部分 机器学习与大模型基础 | Ch11–Ch16 | 从传统机器学习推进到 Tokenizer、Attention 与 Transformer |
| 第三部分 Prompt、Context 与 RAG | Ch17–Ch21 | 构建外部知识增强和可评估的检索生成闭环 |
| 第四部分 Agent 与工程框架 | Ch22–Ch28 | 把模型、工具、状态和人类审批组织成可恢复系统 |
| 第五部分 数据、训练、对齐、评估与安全 | Ch29–Ch39 | 建立从数据到上线门禁的模型能力改造闭环 |
| 第六部分 推理服务与 LLMOps | Ch40–Ch46 | 优化数据面并建设可交付、可观测的生产系统 |
| 第七部分 多模态与前沿架构 | Ch47–Ch53 | 理解多模态、具身和非标准架构的证据边界 |
| 第八部分 岗位与项目面试实战 | Ch54–Ch54 | 把知识转成可验证项目证据和系统设计表达 |

```text
├── 01–10  第一部分 Python 与后端工程基础
├── 11–16  第二部分 机器学习与大模型基础
├── 17–21  第三部分 Prompt、Context 与 RAG
├── 22–28  第四部分 Agent 与工程框架
├── 29–39  第五部分 数据、训练、对齐、评估与安全
├── 40–46  第六部分 推理服务与 LLMOps
├── 47–53  第七部分 多模态与前沿架构
├── 54–54  第八部分 岗位与项目面试实战
```

完整章节表、路线和周计划见 [[00_目录索引]]；旧 40 章如何迁移见 [[docs/RECHAPTERING_MAP]]。

## 代码仓库

`code/` 中有 29 个运行分组、433 个示例，目前覆盖 45/54 个规范章节。目录编号便于批量运行；每个示例的永久身份由 `topic_id` 决定，规范归属见 [`code/TOPIC_MANIFEST.json`](code/TOPIC_MANIFEST.json)。

```powershell
cd code
python -m pip install -r requirements-core.txt
python scripts/verify_all.py
python -m pytest -q -m "not gpu and not slow"
python -m ruff check .
```

运行单个示例：

```powershell
python ch15_transformer/core/01_scaled_dot_product_attention.py
```

批量运行一个分组：

```powershell
python scripts/run_all_examples.py --chapter ch22 --tier llm
```

默认 LLM/GPU 验收使用 mock 或条件跳过，不下载模型、不读取 Key、不产生付费请求。真实 API、GPU、Docker、Redis、pgvector 和模型权重需按脚本 metadata 显式启用。

## 模型与本地数据

模型权重不进入 Git。教程模型默认位于仓库外：

```text
E:\AI_Models\Projects\MyDocument\Python到大模型应用_面试教程_2026版\models
```

下载任何内容优先尝试迅雷；模型权重优先在 ModelScope 查找，没有时再使用 Hugging Face。不要提交 `.env`、API Key、本地缓存或模型文件。

## Obsidian 阅读

将仓库根目录作为 Vault 打开。正文只使用仓库门禁覆盖的 WikiLink、Markdown、MathJax 和 Mermaid 语法；如果修改图、表格或公式，请运行：

```powershell
python code/scripts/verify_all.py
```

该命令检查 WikiLink、章节契约、Markdown/Obsidian 渲染风险、Mermaid、代码路径、来源索引、快照一致性和离线 smoke。

## 质量与贡献

- 章节规范：[[docs/CHAPTER_STYLE_GUIDE]]
- 章节模板：[[docs/CHAPTER_TEMPLATE]]
- 权威来源：[[docs/AUTHORITATIVE_SOURCES]]
- 代码说明：[`code/README.md`](code/README.md)
- 贡献约定：[`CONTRIBUTING.md`](CONTRIBUTING.md)
- 健康报告：[[99_库健康检查报告]]

提交前至少运行 `make ci-quick`、相关 pytest 和 `make lint`。真实 API/GPU 结果要注明版本、硬件、数据、并发和统计口径。

## 版本记录

- **2026-08-05** — 全面重构为 54 章、8 大部分；按 H2/H3 语义迁移正文，引入稳定 `topic_id` 和代码主题清单。
- **2026-08-04** — 统一旧 40 章导航、正文编号与学习闭环，加入 Obsidian 和结构门禁。
- **2026-07-31** — 完成事实、API、安全与可运行性审校，建立权威来源索引。
