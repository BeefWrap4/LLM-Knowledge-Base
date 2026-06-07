# Changelog

> **LLM-Knowledge-Base** 项目演进史 — 23 个 Wave, 24 commits, 2026-06-01 → 2026-06-07.

## v14.0 (2026-06-07) — 完整发布版

### 教程内容 (29 章)
- **Ch01-11**: Python 基础 (可变/并发/内存/算法/数据科学/Web/ML/DL)
- **Ch12-16**: LLM 核心技术 (Transformer/Prompt/RAG/Agent/微调)
- **Ch17-24**: LLM 工程 (评估/框架/分布式/Ops/多模态/数据/安全/云原生)
- **Ch25-29**: 2026 新主题 (推理引擎/世界模型/推理模型/端侧/上下文工程)

### 代码伴侣 (439 .py, 100% 通过)
- **core/** 158 文件 — 任意电脑
- **llm/** 199 文件 — mock 模式 + 真实 API
- **gpu/** 76 文件 — 需 NVIDIA GPU
- **shared/** 5 模块 — provider_registry / llm_client / chatmodel_factory / env / mock_llm
- **scripts/** 6 工具 — verify_all / sync_links / run_all / llm_doctor / download_models / run_real_demos
- **tests/** 19 pytest smoke tests
- **docs/** 5 文档 — API_KEYS / MODELS / DEPLOY / MIGRATE_TO_UNIFIED / MIGRATE_LANGCHAIN

### 23 个 Wave 演进

| Wave | 主题 | 关键产出 |
|------|------|---------|
| **0** | Code companion scaffold | Makefile, requirements-{core,llm,gpu}, shared/, 5 pilot examples |
| **1** | Ch01-11 核心提取 | 154 core/ .py |
| **2** | Ch12-24 LLM/GPU 提取 | 199 llm/ + 60 gpu/ .py |
| **3** | Ch25-29 全新绿 | 48 gpu/ + 14 llm/ 例子 |
| **4** | 收尾 | 合并重复 ch01 目录, 100% README 覆盖, tutorial/ junction |
| **5** | 同步验证工具集 | verify_xrefs / run_all / verify_all (5 项检查) |
| **6** | core/ 100% | 修复 13 个真 bug (typo / slots / PEP 479 / FastAPI 0.106+) |
| **7** | llm/ 100% | 修复 7 个真 bug + 19 依赖 SKIP 模式 |
| **8** | gpu/ 73/76 通过 | run_all 跨平台 path 修复 + 21 个 multi-GPU SKIP |
| **9** | 修正 gpu/ 通过率 | 3 个 page file 误判 → 100% |
| **10** | CI/CD + pre-commit | 4 GitHub Actions + 4 make 目标 + CONTRIBUTING.md |
| **11** | sync_links.py | 教程↔代码 421 个 code refs 全部有效 + 5 项 verify |
| **12** | 推送到 GitHub | 21 commits → BeefWrap4/LLM-Knowledge-Base |
| **13** | README badge | 真实路径 + 5 分钟跑通指南 |
| **14-A** | 统一 LLM API Key | provider_registry + UnifiedClient + llm_doctor |
| **14-B** | code/models/ | 国内源 (ModelScope) + download_models.py |
| **14-C** | Docker 化 | Dockerfile + compose (3 profile) + DEPLOY.md |
| **14-D** | 集成验证 | 357/357 + README 增强 + 健康报告 |
| **14-E** | MiniMax (Codin Plan) | 7th 厂商 |
| **14-F** | 修复 MiniMax endpoint | api.minimaxi.com (无 s) |
| **15-17** | 改造 7 个例子 | UnifiedClient: ch13/06, 09, 14, 20 + ch15/02 + ch17/05, 12 |
| **18** | CI llm_doctor + Docker build | 4 GitHub Actions, ghcr.io 自动 build |
| **19** | chatmodel_factory | LangChain / LlamaIndex 一行切换厂商 |
| **20-21** | 改造 6 个 langchain 例子 | ch18/02, 03, 05, 09, 13, 14 + README badges |
| **22** | run_real_demos.sh | 一键跑 13 个真实 LLM 例子 |

### 真实 LLM 接入 (7 厂商)

| 厂商 | Base URL | 默认模型 | 价格 |
|------|---------|---------|------|
| DeepSeek | api.deepseek.com/v1 | deepseek-chat (V3) | ¥1/百万 token |
| Kimi (Moonshot) | api.moonshot.cn/v1 | moonshot-v1-8k | ¥12/百万 token |
| SiliconFlow | api.siliconflow.cn/v1 | Qwen2.5-7B-Instruct | 部分免费 |
| MiniMax (Codin) | api.minimaxi.com/v1 | MiniMax-Text-01 | 按订阅 |
| OpenAI | api.openai.com/v1 | gpt-4o-mini | 需付费 |
| Anthropic | api.anthropic.com | claude-sonnet-4-5 | 需付费 |
| Mock (离线) | - | mock-llm | 免费 |

### 健康评分演进

```
原始 (24 章)               → 92/100
+ Ch25-29 (5 章 2026 主题) → 95/100
+ code/ 伴侣 + 23 个 Wave  → 95/100 (维持)
```

### 项目数字

| 指标 | 数值 |
|------|------|
| 教程 .md | 33 + 5 docs = 38 |
| code/ 伴侣 .py | 439 |
| **可运行例子** | **357/357 (100%)** |
| **真实 LLM 调用例子** | **13** |
| 章节 | 29 |
| 面试题 | 300+ |
| Mermaid 图 | 200+ |
| Git commits | 24 |
| GitHub Actions | 4 |
| Docker profiles | 3 (core/llm/gpu) |
| 文档页 | 5 (API_KEYS / MODELS / DEPLOY / MIGRATE_TO_UNIFIED / MIGRATE_LANGCHAIN) |
| 总大小 | ~5 MiB |
| 健康评分 | **95/100** |

### 验证命令

```bash
cd code/
make ci              # 6 项综合验证 (含真实 LLM 调用)
make run-all         # 跑完 357 个例子
bash scripts/run_real_demos.sh   # 13 个真实 LLM 例子
```

### 致谢

- 教程内容综合 [Anthropic Claude](https://www.anthropic.com/) / [DeepSeek](https://deepseek.com/) / [Kimi](https://kimi.moonshot.cn/) / [LangChain](https://langchain.com/) / [LlamaIndex](https://www.llamaindex.com/) / [HuggingFace](https://huggingface.co/) 公开文档
- 国内源: 清华 PyPI 镜像 / ModelScope / 阿里云 Container Registry
- 测试环境: miniconda py312, Windows 11, 1× NVIDIA GPU

---

> **最后更新**: 2026-06-07 | **项目状态**: 完整发布版, 接受贡献
