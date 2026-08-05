# Changelog

> **LLM-Knowledge-Base** 项目演进与发布记录。

## Unreleased

### 54 章语义重构

- 将旧 Ch01–Ch40 按学习依赖和交付边界重构为 54 章、8 大部分；265 个旧正文 H2 主题全部且仅迁移一次。
- 拆分 RAG、Agent、对齐、评估、安全治理、推理服务、LLMOps 和多模态等过大章节；为 Tokenizer、Attention、Computer Use、JAX、PD 分离、扩散模型和模型合并建立单一责任章。
- 新增 `docs/RECHAPTERING_MAP.md`，记录旧章到新章的可追溯映射、边界规则和经审查的篇幅例外。

### 代码身份与入口同步

- 保留 29 个运行分组和 433 个示例，将永久归属从易变的章节号迁移为唯一 `topic_id`；新增 `code/TOPIC_MANIFEST.json` 作为规范映射。
- 同步 README、目录、思维导图、权威来源、示例 metadata、代码 README 和测试；新增幂等的目录文档与主题清单生成器。
- 验收门禁改为验证 54 章、稳定主题 ID、代码路径、Obsidian/Markdown/Mermaid 结构、权威来源和入口文档快照。

### 章节叙事与学习闭环

- 全量统一 Ch01–Ch40 的 H1、章首导航、三项可验收学习目标、连续 H2 编号和固定章末栏目；
  保留原技术正文，将散落或重复的小结、练习、代码、面试题、速查、相关章节和来源归入统一顺序。
- 章首移除无法复核的固定面试占比与宣传式措辞，保留版本边界、适用岗位和证据日期；缺少导入的章节补上
  问题定位与阅读方法，使学习路径从概念、实践推进到失败边界和面试表达。
- 新增 `docs/CHAPTER_STYLE_GUIDE.md`、`docs/CHAPTER_TEMPLATE.md` 和幂等规范化脚本；结构门禁检查
  唯一 H1、导航位置、三项目标、导入段、正文编号、固定结尾的完整性与顺序。

### Obsidian 与 Markdown 兼容性

- 修复 7 个在 Obsidian 1.12.7 / Mermaid 11.4.1 中无法解析的图表，包括非法 `tree`
  声明、节点闭合符错误，以及未加引号的 `[MASK]`、数学花括号和函数圆括号标签。
- 修复 15 个可解析但无法正确渲染的图表，避免数字序号、单独 `+` 和前导 `>` 被
  Mermaid 的 Markdown 词法器误判为不受支持的列表或引用块。
- 为 261 个 Mermaid 代码块增加 fail-closed 结构门禁，阻止未知图表类型、未闭合代码围栏和
  常见的节点标签解析冲突及不受支持 Markdown 标签再次进入主分支。
- 全量审计 101 份 Obsidian 可见维护文档，修复 4 个错列表格、2 个会被内部三反引号提前
  终止的 Markdown 示例、1 处标题层级跳跃和 1 处未转义的价格单位写法。
- 将 29 个代码章节 README 的教程入口统一为仓库内直接相对路径，移除对本机
  `code/tutorial` 符号链接的渲染依赖。
- 新增 Markdown 渲染门禁，覆盖代码围栏、数学定界符、标题层级与重复标题、表格列数、
  WikiLinks、本地链接、Callout、YAML Frontmatter、HTML 注释/容器和符号链接路径。
- 修复 SVM 核函数表被范数竖线误拆为 7 列的问题，并将 5 行表格中的范数统一改为
  `\lVert ... \rVert`；门禁新增逐单元格数学定界符与多空表头检查，防止同类回归。
- 全量语义与布局审计 261 个 Mermaid 图，修复 14 个角色、数据流或超宽布局问题，覆盖 ReAct 工具调用、
  MCP Host/Client/Server、审计链路、Prometheus 抓取、Kubernetes 控制面、OpenTelemetry
  导出链路、模型网关和 Context Builder；新增 ReAct Action/Observation 路由门禁。

## v1.1.0 (2026-07-31) — 40 章审校与 fail-closed 验收

### 教程与来源

- 正文扩展并统一为 Ch01–Ch40；代码伴侣保持 29 章、433 个示例。
- 对模型/API/框架/安全/国内岗位材料进行时效性与准确性审校。
- 新增 40/40 章权威来源台账；README、目录、思维导图与健康报告统计一致。

### 验证与安全

- 十阶段离线门禁覆盖结构、WikiLinks、来源、快照、代码引用和代表性 smoke。
- 禁止示例使用内置 `eval`/`exec`；真实 API 需双重确认，LLM runner 强制离线。
- GPU 外部副作用按下载、服务、端口、编译、浏览器、云和平台分别显式授权。
- 本地回归：304 non-GPU pytest PASS、3 GPU pytest PASS、Ruff/compileall PASS。
- 433 个示例离线终态：365 PASS、68 SKIP、0 FAIL。

### 条件性实测与修复

- RTX 5090 D：76 个 GPU 示例按章串行，47 PASS、29 条件 SKIP、0 FAIL。
- DeepSeek 最小真实冒烟与 bge/Redis/pgvector/DeepSeek 四组件集成通过。
- Docker 改用官方 PyPI 默认源，镜像包含完整教程验收输入；容器内十阶段门禁通过。
- Docker Actions 升级到 Node.js 24 运行时主版本，消除 GitHub Runner 的 Node.js 20 弃用告警。
- Compose 支持隔离项目名，补齐 pgvector `15432` 映射及可配置 Redis/Postgres 连接。
- 修复 Flash Attention naive/SDPA causal 语义不一致及多个示例缺失 `OK` 契约。

> `v1.1.0` 标签已在远端 Verify 与 Docker/GHCR 工作流全部通过后发布。

## Legacy Wave 14.0 snapshot (2026-06-07)

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
