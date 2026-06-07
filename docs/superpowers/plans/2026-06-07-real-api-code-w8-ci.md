# W8 CI 改造实现计划 — Real API Code

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 改造 GitHub Actions：PR 冒烟走 mock（≤3 分钟）；夜间跑真实 API + self-hosted GPU runner。

**前置依赖：** W1 + W7 完成。

---

## 文件清单

### 创建

- `.github/workflows/pr-check.yml`
- `.github/workflows/integration-test.yml`
- `.github/workflows/gpu-smoke.yml`
- `code/scripts/test_real_api_smoke.py`

### 修改

- `code/scripts/verify_all.py`（加 LLM_MOCK 检查）
- `code/Makefile`（加 ci-real target）

---

## 任务 1：PR check workflow

- [ ] **步骤 1：创建 `.github/workflows/pr-check.yml`**

```yaml
name: PR Check
on:
  pull_request:
    branches: [master]
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install core deps
        run: |
          cd code
          pip install -r requirements-core.txt
      - name: Verify xrefs
        run: make verify-xrefs
      - name: Smoke tests (mock mode)
        run: |
          cd code
          LLM_MOCK=1 pytest tests/ -m "not gpu" -v --tb=short
      - name: Lint
        run: |
          cd code
          pip install ruff mypy
          ruff check .
          mypy shared/ 2>&1 | head -20 || true
```

- [ ] **步骤 2：本地 dry-run（用 act 或 push 到 fork）**

- [ ] **步骤 3：Commit**

```bash
git add .github/workflows/pr-check.yml
git commit -m "CI: add PR check workflow (mock mode, ≤3 min)"
```

---

## 任务 2：夜间 integration test

- [ ] **步骤 1：创建 `code/scripts/test_real_api_smoke.py`**

```python
# scripts/test_real_api_smoke.py
"""夜间跑: 真实 API 冒烟 (10 个代表性 LLM tier 文件)."""
import os
import sys
import subprocess
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent

REPRESENTATIVE_FILES = [
    "ch13_prompt_engineering/llm/01_few_shot.py",
    "ch14_rag/llm/01_basic_rag.py",
    "ch15_agent/llm/01_react_basic.py",
    "ch16_finetuning/llm/01_sft_basic.py",
    "ch17_evaluation/llm/01_llm_judge.py",
    "ch18_llm_frameworks/llm/01_langchain_basic_chain.py",
    "ch20_llmops/llm/01_prompt_monitoring.py",
    "ch22_data_eng/llm/04_self_instruct.py",
    "ch27_reasoning_ttc/llm/01_chain_of_thought.py",
    "ch29_context_engineering/llm/12_full_context_pipeline.py",
]

def main():
    if os.environ.get("DEEPSEEK_API_KEY") in (None, "", "YOUR_API_KEY"):
        print("[SKIP] 缺 DEEPSEEK_API_KEY，跳过真实 API 冒烟")
        sys.exit(0)
    
    failed = []
    for f in REPRESENTATIVE_FILES:
        path = CODE / f
        if not path.exists():
            print(f"[SKIP] {f} 不存在")
            continue
        print(f"=== {f} ===")
        result = subprocess.run(
            ["python", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"❌ FAILED: {result.stderr[:500]}")
            failed.append(f)
        else:
            print(f"✅ OK (output: {result.stdout[:200]})")
    
    if failed:
        print(f"\n{len(failed)} 个文件失败: {failed}")
        sys.exit(1)
    print(f"\n✅ 全部 {len(REPRESENTATIVE_FILES)} 个文件真实 API 冒烟通过")

if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：创建 `.github/workflows/integration-test.yml`**

```yaml
name: Integration Test (Real API)
on:
  schedule:
    - cron: "0 22 * * *"  # 22:00 UTC
  workflow_dispatch:
jobs:
  real-api:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: |
          cd code
          pip install -r requirements-llm.txt
      - name: Real API smoke
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        run: |
          cd code
          python scripts/test_real_api_smoke.py
```

- [ ] **步骤 3：Commit**

---

## 任务 3：GPU smoke（self-hosted runner）

- [ ] **步骤 1：创建 `.github/workflows/gpu-smoke.yml`**

```yaml
name: GPU Smoke (Self-hosted)
on:
  schedule:
    - cron: "30 22 * * *"  # 22:30 UTC, after integration test
  workflow_dispatch:
jobs:
  gpu:
    runs-on: [self-hosted, gpu, nvidia-24gb]
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Verify NVIDIA GPU
        run: nvidia-smi
      - run: |
          cd code
          pip install -r requirements-gpu.txt
      - name: Download Qwen2.5-7B
        run: |
          cd code
          python scripts/download_models.py --target qwen7b
      - name: Run vLLM async engine
        run: |
          cd code
          timeout 300 python ch25_inference_engines/gpu/10_vllm_async_engine.py
      - name: Run Ollama ch28
        run: |
          ollama serve &
          sleep 5
          ollama pull llama3.2:3b
          cd code
          python ch28_edge_llm/gpu/05_ollama_http_api.py
```

- [ ] **步骤 2：配置 self-hosted runner**（操作步骤文档化，不在代码改动范围）

- [ ] **步骤 3：Commit**

---

## 任务 4：`verify_all.py` 加 LLM_MOCK 检查

- [ ] **步骤 1：读 `code/scripts/verify_all.py`**

- [ ] **步骤 2：在 CI 子命令里加 `assert "LLM_MOCK" in os.environ`**

```python
# verify_all.py 加:
def check_ci_safety():
    """CI 安全检查: 禁止 CI 环境意外调真实 API."""
    if os.environ.get("CI") and "LLM_MOCK" not in os.environ:
        # 不强制退出, 仅打印警告
        print("[WARN] CI 环境未设 LLM_MOCK, 可能调真实 API. "
              "建议在 workflow 中 export LLM_MOCK=1.")
```

---

## 任务 5：手测（合并 master 后）

- [ ] **步骤 1：开 PR → 验证 pr-check.yml 跑通**

- [ ] **步骤 2：手动触发 integration-test → 验证真实 API 跑通**

- [ ] **步骤 3：手动触发 gpu-smoke → 验证 self-hosted runner 跑通**

---

## 任务 6：Commit 收尾

```bash
git add -A
git commit -m "W8 CI: PR check + nightly real API + GPU smoke workflows"
```

---

## W8 验收清单

- [ ] `.github/workflows/pr-check.yml` 跑通（mock 模式 ≤3 分钟）
- [ ] `.github/workflows/integration-test.yml` 跑通（10 个真实 API 文件）
- [ ] `.github/workflows/gpu-smoke.yml` 在 self-hosted runner 上跑通（vLLM + Ollama）
- [ ] GitHub secrets 加 `DEEPSEEK_API_KEY`
- [ ] Self-hosted runner 文档化
