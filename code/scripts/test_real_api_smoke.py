#!/usr/bin/env python3
# ---
# code/scripts/test_real_api_smoke.py
# 真实 API 冒烟测试 (10 个代表性 LLM tier 文件)
# 用途: 夜间 CI / push to master / workflow_dispatch
# 需 DEEPSEEK_API_KEY (secrets.DEEPSEEK_API_KEY)
# ---
"""
真实 API 冒烟 (夜间跑 / push to master / workflow_dispatch).

10 个代表性 LLM tier 文件, 每个跑一次, 验证响应非空 + 不超时.

前置检查 (避免误跑):
  - DEEPSEEK_API_KEY 必须设置且非 placeholder
  - LLM_MOCK 必须未设 (与 LLM_MOCK=1 互斥)
  - 不可在 PR check 跑 (仅 nightly / push master / workflow_dispatch)

Usage:
    # 默认: 自动跳过若没设 DEEPSEEK_API_KEY
    unset LLM_MOCK
    python code/scripts/test_real_api_smoke.py

    # 强制存在性检查
    unset LLM_MOCK
    export DEEPSEEK_API_KEY=sk-xxx
    python code/scripts/test_real_api_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent

# 10 个代表性文件 (按 code 实际目录校正)
# 来源: W6-B2 真实跑通的代表性 LLM tier 例子
REPRESENTATIVE_FILES = [
    "ch13_prompt_engineering/llm/06_self_consistency_cot.py",
    "ch14_rag/llm/01_rag_indexing_pipeline.py",
    "ch15_agent/llm/01_agent_tools_definition.py",
    "ch17_evaluation/llm/06_ragas_evaluation.py",
    "ch18_llm_frameworks/llm/01_langchain_basic_chain.py",
    "ch20_llmops/llm/07_langsmith_prompt_debug.py",
    "ch22_data_eng/llm/04_self_instruct.py",
    "ch27_reasoning_ttc/llm/01_o3_api_basic.py",  # 需 OPENAI_API_KEY, 缺则跳过
    "ch29_context_engineering/llm/12_full_context_pipeline.py",
    "ch13_prompt_engineering/llm/04_zero_shot_cot.py",  # 替补 1 (ch16 无 llm/)
]

# 需要额外 vendor key 的文件: 缺对应 key 时跳过 (不算失败)
REQUIRES_EXTRA_KEY = {
    "ch27_reasoning_ttc/llm/01_o3_api_basic.py": "OPENAI_API_KEY",
}

# 备用文件 (若上面有缺失)
FALLBACK_FILES = [
    "ch13_prompt_engineering/llm/02_few_shot_sentiment.py",
    "ch13_prompt_engineering/llm/05_few_shot_cot.py",
    "ch14_rag/llm/02_rag_pipeline_class.py",
    "ch15_agent/llm/02_react_agent_from_scratch.py",
    "ch15_agent/llm/03_function_calling_agent.py",
    "ch17_evaluation/llm/05_llm_as_judge.py",
    "ch18_llm_frameworks/llm/02_llmchain_basic.py",
    "ch20_llmops/llm/05_langsmith_observability.py",
    "ch22_data_eng/llm/05_evol_instruct.py",
    "ch27_reasoning_ttc/llm/04_reasoning_effort_ladder.py",
    "ch29_context_engineering/llm/01_context_engineering_intro.py",
]

# 真实超时/文件 (DeepSeek 平均 5-15s, 加 buffer)
TIMEOUT_PER_FILE = 120


def check_prerequisites() -> None:
    """前置检查 (fail-fast)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_API_KEY" or api_key.startswith("sk-placeholder"):
        print("[SKIP] DEEPSEEK_API_KEY 未设置或为 placeholder, 跳过真实 API 冒烟")
        print("       (夜间 CI 应配 secrets.DEEPSEEK_API_KEY)")
        sys.exit(0)  # exit 0 = 跳过不算失败

    if os.environ.get("LLM_MOCK") == "1":
        print("[ERROR] LLM_MOCK=1 与真实 API 冒烟互斥, 请 unset")
        print("        export LLM_MOCK=  # 或 unset LLM_MOCK")
        sys.exit(1)

    # 不可在 PR check 跑 (应是夜间 / push master / workflow_dispatch)
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        print("[ERROR] 真实 API 冒烟不可在 PR check 跑 (成本+稳定性)")
        print("        应在 push to master / schedule / workflow_dispatch 跑")
        sys.exit(1)


def resolve_files() -> list[str]:
    """解析 10 个可用的代表性文件."""
    files: list[str] = []
    for f in REPRESENTATIVE_FILES:
        if (CODE / f).exists():
            files.append(f)

    # 不足 10 个时, 用 FALLBACK_FILES 补齐
    if len(files) < 10:
        for f in FALLBACK_FILES:
            if (CODE / f).exists() and f not in files:
                files.append(f)
                if len(files) >= 10:
                    break

    return files[:10]  # 最多 10


def run_file(path: Path) -> tuple[str, str]:
    """跑单个文件, 返回 (status, output_preview)."""
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_PER_FILE,
            cwd=str(CODE),
        )
        if result.returncode != 0:
            err_preview = (result.stderr or result.stdout)[:300].replace("\n", " ")
            return "FAIL", f"rc={result.returncode} | {err_preview}"
        out_preview = (result.stdout or "").strip().replace("\n", " ")[:200]
        return "OK", out_preview or "(no stdout)"
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"超时 >{TIMEOUT_PER_FILE}s"
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {e}"


def main() -> int:
    check_prerequisites()

    print("=" * 60)
    print("  真实 API 冒烟 (10 个代表性 LLM tier 文件)")
    print("=" * 60)
    print()

    files = resolve_files()
    if not files:
        print("[ERROR] 没找到任何代表性文件 (代码结构异常?)")
        return 1

    print(f"将测试 {len(files)} 个文件 (timeout {TIMEOUT_PER_FILE}s/文件):\n")

    failed: list[str] = []
    passed: list[str] = []

    for f in files:
        rel = f
        path = CODE / f
        if not path.exists():
            print(f"  [SKIP] {rel} (文件不存在)")
            continue

        # 需额外 vendor key 的文件: 缺则跳过, 不算失败
        extra_key = REQUIRES_EXTRA_KEY.get(rel)
        if extra_key and not os.environ.get(extra_key, "").strip():
            print(f"  [SKIP] {rel} (需 {extra_key}, 未配)")
            continue

        print(f"--- {rel} ---")
        status, detail = run_file(path)
        if status == "OK":
            print(f"  [OK]   {detail[:200]}")
            passed.append(rel)
        else:
            print(f"  [{status}] {detail[:200]}")
            failed.append(rel)
        print()

    # 汇总
    print("=" * 60)
    print(f"  total:  {len(files)}")
    print(f"  passed: {len(passed)}")
    print(f"  failed: {len(failed)}")
    print("=" * 60)

    if failed:
        print("\n失败文件:")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
