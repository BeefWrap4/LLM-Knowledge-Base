# W7 教程同步实现计划 — Real API Code

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 同步 29 个 .md 教程与 W3-W6 改造后的代码；新增 README 段；保证反向链接全通。

**前置依赖：** W3-W6 全部完成。

---

## 文件清单

### 重写（实质性更新）

- `00_目录索引.md`
- `17_大模型评估体系.md`
- `18_LLM工程框架实战.md`
- `25_推理引擎与高性能服务.md`
- `26_世界模型与具身AI.md`
- `27_推理模型与Test-Time_Compute.md`
- `28_端侧与边缘LLM.md`

### 小改（替换"运行命令"段）

- `13_Prompt_Engineering.md`
- `14_RAG检索增强生成.md`
- `15_Agent智能体开发.md`
- `16_模型微调与推理优化.md`
- `20_LLMOps与模型可观测性.md`
- `22_大模型数据工程.md`
- `29_Context_Engineering.md`

### README

- `README.md`
- `code/README.md`

---

## 任务 1：`00_目录索引.md` — 加"硬件 × 章节矩阵"表

- [ ] **步骤 1：在 "板块架构"段后追加"硬件 × 章节矩阵"表（参考 W1 task 9）**

- [ ] **步骤 2：在"快速开始"段加 `make llm-doctor-setup` 命令**

---

## 任务 2：`18_LLM工程框架实战.md` — 删 mock demo 段

- [ ] **步骤 1：grep 找 "mock" / "fake" / "FakeListChatModel"**

- [ ] **步骤 2：替换为真实调用示例 + LLM_MOCK=1 提示**

---

## 任务 3：`25_推理引擎与高性能服务.md` — 加 vLLM 真实启动段

- [ ] **步骤 1：在 §25.2 vLLM 章节加 `make download-models-llm` + 真实启动命令**

- [ ] **步骤 2：加硬件需求小节（NVIDIA 24GB+）**

---

## 任务 4：`26_世界模型与具身AI.md` — 加 Cosmos/Pi0 真实加载

---

## 任务 5：`27_推理模型与Test-Time_Compute.md` — 加 R1-Distill 真实运行

---

## 任务 6：`28_端侧与边缘LLM.md` — 加 Ollama 真实调用 + MLX 真实段

---

## 任务 7：13, 14, 15, 16, 20, 22, 29 替换"运行命令"段

- [ ] **步骤 1：grep "运行例子" 段**

- [ ] **步骤 2：替换为 `export DEEPSEEK_API_KEY=...` + 真实命令**

---

## 任务 8：`code/README.md` 已更新（W1），审一遍

---

## 任务 9：`README.md` — 加"配套代码使用指南"段

---

## 任务 10：跑 `make verify-xrefs` 全绿

```bash
cd code
make verify-xrefs
make verify
```

预期：所有反向链接 200，全文 4 项检查通过。

---

## 任务 11：Commit 收尾

```bash
git add -A
git commit -m "W7 tutorial sync: 29 .md updated, README hardware matrix, xrefs verified"
```

---

## W7 验收清单

- [ ] 7 个重灾章节实质性更新
- [ ] 7 个小改章节替换运行命令
- [ ] README / code/README 加硬件矩阵
- [ ] `make verify-xrefs` 全绿
- [ ] `make verify` 4 项检查全绿
