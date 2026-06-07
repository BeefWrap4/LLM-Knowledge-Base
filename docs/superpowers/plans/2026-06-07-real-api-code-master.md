# Real API / Real Model / Real Framework — 实现计划（Master）

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `code/` 目录 450 个 .py 文件从 mock / fake 改造为真实 API + 真实模型 + 真实框架；主流程不再依赖 mock；mock 路径下沉为 CI 专用；教程 markdown 与代码完全同步。

**架构：** 4 层架构（教程 / 代码 / 共享 / 外部服务），8 个分阶段 wave，每 wave 独立 PR，按风险从低到高（core → llm → gpu）推进。

**技术栈：** Python 3.11+、openai SDK、LangChain / LlamaIndex / Haystack、vLLM / SGLang / TRT-LLM、MLX、Ollama、DeepSeek / Kimi / SiliconFlow / MiniMax / OpenAI / Anthropic API。

**关联设计文档：** `docs/superpowers/specs/2026-06-07-real-api-code-design.md`

---

## 1. Wave 切分（独立可交付）

| 计划文件 | Wave | 范围 | 周期 | 风险 |
|---------|------|------|------|------|
| [`2026-06-07-real-api-code-w1-infrastructure.md`](./2026-06-07-real-api-code-w1-infrastructure.md) | W1 基建 | `shared/` + `scripts/` + `Makefile` + `tests/_mocks/` + `code/README.md` | 3-5 天 | 🟢 低 |
| [`2026-06-07-real-api-code-w2-core.md`](./2026-06-07-real-api-code-w2-core.md) | W2 Core tier | 158 个 `core/*.py` 审计 | 1-2 天 | 🟢 低 |
| [`2026-06-07-real-api-code-w3-llm.md`](./2026-06-07-real-api-code-w3-llm.md) | W3 LLM tier | 199 个 `llm/*.py` + 29 mock import | 5-7 天 | 🟡 中 |
| [`2026-06-07-real-api-code-w4-edge-gpu.md`](./2026-06-07-real-api-code-w4-edge-gpu.md) | W4 端侧 GPU | `ch28/.../gpu/` 10 个 | 2-3 天 | 🟡 中 |
| [`2026-06-07-real-api-code-w5-inference-gpu.md`](./2026-06-07-real-api-code-w5-inference-gpu.md) | W5 推理引擎 GPU | `ch25/.../gpu/` 12 个 | 3-4 天 | 🔴 高 |
| [`2026-06-07-real-api-code-w6-training-world.md`](./2026-06-07-real-api-code-w6-training-world.md) | W6 训练/世界模型 | `ch19/ch26/ch27/ch16` GPU 30 个 | 4-5 天 | 🔴 高 |
| [`2026-06-07-real-api-code-w7-tutorial-sync.md`](./2026-06-07-real-api-code-w7-tutorial-sync.md) | W7 教程同步 | 29 个 .md + README | 5-7 天 | 🟡 中 |
| [`2026-06-07-real-api-code-w8-ci.md`](./2026-06-07-real-api-code-w8-ci.md) | W8 CI 改造 | `.github/workflows/` 4 个 | 2-3 天 | 🟢 低 |

**总计**：~200 个文件改动，25-36 天。

---

## 2. 跨 wave 全局约束（所有 wave 都要遵守）

### 2.1 主流程不变量

- L2 主流程 `.py` 文件**禁止**出现 `is_mock / provider == "mock" / if MOCK` 等条件分支
- L2 唯一允许的环境变量开关是：`if os.environ.get("LLM_MOCK") == "1": ...`，且必须紧跟 `sys.exit(0)`
- 任何 commit 出现违规，CI 拒绝

### 2.2 错误处理统一出口

- 缺 API Key / 缺模型权重 / 缺硬件 → 抛 `RuntimeError` 带明确信息
- 抛错信息统一格式：`[ERROR] {file}:{line}  {异常类}: {消息}\n[HELP]  参考 README#硬件矩阵`
- 使用 `shared/_error_helper.py:format_error()`（W1 创建）

### 2.3 Mock 路径边界

- `shared/mock_llm.py` → `tests/_mocks/mock_llm.py`（W3 一次性迁移）
- `shared/__init__.py` 不再导出 `MockLLM / deterministic_response`
- `tests/_mocks/conftest.py` 在 pytest 启动时设 `LLM_MOCK=1`

### 2.4 CI 行为

- PR 检查 ≤ 3 分钟：mock 路径 + 静态检查
- 夜间集成测试：真实 API + self-hosted GPU runner
- 详情见 W8 计划

---

## 3. 全局文件结构（所有 wave 完成后）

```
D:\MyDocument\Python到大模型应用_面试教程_2026版_分章节\
├── docs/
│   └── superpowers/
│       ├── specs/
│       │   └── 2026-06-07-real-api-code-design.md       (已存在)
│       └── plans/
│           ├── 2026-06-07-real-api-code-master.md       (本文件)
│           ├── 2026-06-07-real-api-code-w1-infrastructure.md
│           ├── 2026-06-07-real-api-code-w2-core.md
│           ├── 2026-06-07-real-api-code-w3-llm.md
│           ├── 2026-06-07-real-api-code-w4-edge-gpu.md
│           ├── 2026-06-07-real-api-code-w5-inference-gpu.md
│           ├── 2026-06-07-real-api-code-w6-training-world.md
│           ├── 2026-06-07-real-api-code-w7-tutorial-sync.md
│           └── 2026-06-07-real-api-code-w8-ci.md
├── code/                            (主代码)
│   ├── shared/
│   │   ├── llm_client.py            (W1 改造)
│   │   ├── provider_registry.py     (W1 改造)
│   │   ├── gpu_guard.py             (W1 扩展)
│   │   ├── _error_helper.py         (W1 新增)
│   │   ├── chatmodel_factory.py     (W1 审)
│   │   └── __init__.py              (W1 改导出)
│   ├── scripts/
│   │   ├── llm_doctor.py            (W1 加 --setup)
│   │   ├── download_models.py       (W1 扩模型清单)
│   │   ├── verify_all.py            (W8 改)
│   │   └── run_all_examples.py      (W1 审)
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_pilots.py
│   │   ├── test_shared.py
│   │   ├── _mocks/                  (W1 新建)
│   │   │   ├── __init__.py
│   │   │   ├── conftest.py          (自动 LLM_MOCK=1)
│   │   │   ├── mock_llm.py          (← shared/mock_llm.py)
│   │   │   ├── mock_embedding.py    (← ch18 _MockEmbed)
│   │   │   ├── mock_vllm.py         (← ch25/10)
│   │   │   └── mock_trt.py          (← ch25/08)
│   │   └── test_real_api_smoke.py   (W8 新建)
│   ├── models/                      (W1 扩)
│   │   ├── bge-small-zh-v1.5/       (已有)
│   │   ├── bge-reranker-v2-m3/      (已有)
│   │   ├── Qwen2.5-0.5B-Instruct/   (已有)
│   │   ├── Qwen2.5-7B-Instruct/     (W1 准备)
│   │   ├── Llama-3.1-8B-Instruct/   (W1 准备)
│   │   ├── Cosmos-1.0-7B/           (W1 准备)
│   │   ├── Pi0-VLA-base/            (W1 准备)
│   │   └── DeepSeek-R1-Distill-Qwen-1.5B/  (W1 准备)
│   ├── ch01-29/                     (W2-W6 改)
│   └── README.md                    (W1 加硬件矩阵段)
├── tutorial/                        (W7 改 — 软链接到 18_*.md 等)
├── .github/
│   └── workflows/
│       ├── pr-check.yml             (W8 新建)
│       ├── integration-test.yml     (W8 新建)
│       └── gpu-smoke.yml            (W8 新建)
└── 18_LLM工程框架实战.md 等 29 个 .md  (W7 改)
```

---

## 4. 执行顺序与依赖

```
W1 (基建) ──┬──> W2 (core 审)
            ├──> W3 (llm 真)
            ├──> W4 (端侧 GPU)
            ├──> W5 (推理引擎)
            └──> W6 (训练/世界模型)
                │
                v
            W7 (教程同步)        ← 依赖 W3-W6 全部完成
                │
                v
            W8 (CI 改造)         ← 依赖 W1-W7
```

**W1 是其他所有 wave 的硬依赖**。  
**W2-W6 之间无依赖，可并行（不同 worktree）**。  
**W7 依赖 W3-W6 完成**。  
**W8 依赖 W1 + W7**。

---

## 5. 验收（Wave 全部完成后）

满足以下全部条件，视为本次重构完成：

- [ ] `grep -rln "is_mock\|fake_llm\|MockLLM()" code/` 输出 0 行
- [ ] `python xx.py` 在 4 类用户环境下全部跑通：
  - [ ] 有 `DEEPSEEK_API_KEY` 的国内用户
  - [ ] Apple M-series + Ollama 已启动
  - [ ] NVIDIA 24GB+ + vLLM 权重已下载
  - [ ] 无 Key + `LLM_MOCK=1`（CI / 离线）
- [ ] `make ci-core` / `make ci-llm` (mock) / 夜间 integration test 全绿
- [ ] 29 个 .md 教程反向链接全通（`make verify-xrefs`）
- [ ] 仓库 README "硬件 × 章节矩阵"表完整

---

## 6. 不在范围内

- ❌ 改教程非代码章节（面试题答案、概念解释）
- ❌ 添加新教程章节
- ❌ 替换 mock_llm 的实现
- ❌ 引入新的 LLM 厂商
- ❌ 改测试策略为端到端
