---
title: GitHub Actions Secrets 配置指南
created: 2026-06-08
tags: [ci, secrets, github-actions]
---

# GitHub Actions Secrets 配置指南

本仓库的 CI workflow 需要多个 LLM API key 作为 secrets。**push 后 workflow 会用空 key 跑并优雅跳过**，但要真跑出绿色结果，必须在 GitHub 仓库 settings 配置 secrets。

## 配置步骤

1. 打开 https://github.com/BeefWrap4/LLM-Knowledge-Base/settings/secrets/actions
2. 点 **"New repository secret"**
3. 按下表添加每个 key

## Secrets 清单

| Secret 名 | 必填? | 哪个 workflow 用 | 获取地址 |
|----------|------|----------------|---------|
| `DEEPSEEK_API_KEY` | 推荐 | ci-llm-doctor, integration-test | https://platform.deepseek.com (国内, 注册送 ¥10) |
| `OPENAI_API_KEY` | 可选 | ci-llm-doctor (O3) | https://platform.openai.com |
| `ANTHROPIC_API_KEY` | 可选 | ci-llm-doctor (Claude) | https://console.anthropic.com |
| `KIMI_API_KEY` | 可选 | ci-llm-doctor, integration-test | https://platform.moonshot.cn |
| `SILICONFLOW_API_KEY` | 可选 | ci-llm-doctor, integration-test | https://cloud.siliconflow.cn |
| `MINIMAX_API_KEY` | 可选 | ci-llm-doctor, integration-test | https://api.minimaxi.com |

## 哪些 workflow 需要哪些 key

- `verify.yml` (PR check, 每次提交) — **无 secrets 依赖** ✅
- `ci-llm-doctor.yml` (周一 06:00 UTC + 手动) — 6 个 key **全部可选**，无 key 时 `exit 0` 跳过
- `integration-test.yml` (PR + master + 每周日) — 推荐至少 `DEEPSEEK_API_KEY`
- `docker-build.yml` (push to master + tag v*) — 仅需 `GITHUB_TOKEN` (GitHub 自动注入) ✅
- `gpu-verify.yml` (手动) — **无 secrets 依赖**，但需 self-hosted runner (NVIDIA GPU)

## 验证

配完后,手动触发 `ci-llm-doctor.yml`:
- https://github.com/BeefWrap4/LLM-Knowledge-Base/actions/workflows/ci-llm-doctor.yml
- 点 **"Run workflow"** → 选 master → **"Run workflow"**
- 1 分钟后看每个 provider 状态: ✅ OK / ❌ FAIL (401 错等)

## Self-hosted Runner（Ch40–Ch49 GPU 与推理测试）

`gpu-verify.yml` 需 self-hosted runner:
- 仓库管理员: https://github.com/BeefWrap4/LLM-Knowledge-Base/settings/actions/runners/new
- 选 OS (Linux 推荐), 按 GitHub 文档在带 NVIDIA GPU 的机器跑注册命令
- 标签: `self-hosted, gpu, nvidia` (yml 已配置)

**当前仓库无 self-hosted runner**, 手动触发 `gpu-verify.yml` 会永久 PENDING。

## 故障排除

| 现象 | 原因 | 修复 |
|------|------|------|
| `verify.yml` PR check 红 | 教程 .md 反向链接断 | `make verify-xrefs` 跑通后再 push |
| `integration-test.yml` 红 + `kimi 401` | KIMI_API_KEY 错/过期 | 重新登录 https://platform.moonshot.cn 拿新 key |
| `docker-build.yml` 失败 | GHCR push 权限 | Settings → Actions → General → "Workflow permissions" 选 "Read and write permissions" |
| `gpu-verify.yml` PENDING | 无 self-hosted runner | 见上 |
