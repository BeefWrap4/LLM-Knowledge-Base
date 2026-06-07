# ---
# chapter: 24
# topic: 云原生部署与工程化
# section: 24.6.3 自动化测试与评估门禁
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: (stdlib only)
# run: python 04_eval_gate.py --eval-dataset dummy.jsonl --output eval-results.json
# expected_runtime: <1s
# expected_output: prints "EVALUATION GATE PASSED" and writes eval-results.json
# ---
# See: ../tutorial/24_云原生部署与工程化.md §24.6.3
# Interview hooks:
#   1. 大模型 CI 与传统软件 CI 的最大区别是什么？评估门禁应包含哪些维度？
#   2. 如何设计可回退的评估门禁（避免单次抖动阻塞 PR）？
#   3. accuracy / safety / latency / token_error_rate 这四个指标为什么缺一不可？


# === Multi-GPU / heavy model guard (auto-added) ===
import sys as _sys
import os as _os
_NGPU = _os.environ.get("WORLD_SIZE", "1")
if _NGPU == "1" and not _os.environ.get("FORCE_GPU_RUN"):
    print(f"[SKIP] {{__file__}}: 需多卡 (WORLD_SIZE>1) 或真实模型权重, 用 torchrun 或设置 FORCE_GPU_RUN=1")
    _sys.exit(0)
"""
模型评估门禁框架 —— 在 CI 中自动执行的评估脚本
"""

import json
import sys
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class EvalGateConfig:
    """评估门禁配置"""
    accuracy_threshold: float = 0.85
    safety_threshold: float = 0.99
    latency_p99_threshold_ms: float = 5000.0
    token_error_rate_threshold: float = 0.05

    def check(self, results: dict) -> List[str]:
        """检查评估结果是否通过所有门禁"""
        failures = []

        if results.get("accuracy", 0) < self.accuracy_threshold:
            failures.append(
                f"Accuracy {results['accuracy']:.3f} < {self.accuracy_threshold}"
            )
        if results.get("safety_score", 0) < self.safety_threshold:
            failures.append(
                f"Safety {results['safety_score']:.3f} < {self.safety_threshold}"
            )
        if results.get("latency_p99_ms", 0) > self.latency_p99_threshold_ms:
            failures.append(
                f"Latency P99 {results['latency_p99_ms']}ms > {self.latency_p99_threshold_ms}ms"
            )
        if results.get("token_error_rate", 0) > self.token_error_rate_threshold:
            failures.append(
                f"Token Error Rate {results['token_error_rate']:.3f} > {self.token_error_rate_threshold}"
            )
        return failures


def run_evaluation(eval_dataset_path: str) -> dict:
    """
    执行模型评估（简化示例）
    实际应用会调用 LLM-as-Judge、评估框架（如 lm-evaluation-harness）等
    """
    # 模拟评估结果
    results = {
        "accuracy": 0.872,
        "safety_score": 0.995,
        "latency_p99_ms": 4200.0,
        "latency_p50_ms": 1850.0,
        "token_error_rate": 0.012,
        "bleu_score": 0.34,
        "rouge_l": 0.52,
        "hallucination_rate": 0.03,
        "total_test_cases": 1000,
        "timestamp": "2026-06-01T10:00:00Z",
    }
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--threshold-accuracy", type=float, default=0.85)
    parser.add_argument("--threshold-safety", type=float, default=0.99)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # 执行评估
    results = run_evaluation(args.eval_dataset)

    # 写入结果
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # 门禁检查
    config = EvalGateConfig(
        accuracy_threshold=args.threshold_accuracy,
        safety_threshold=args.threshold_safety,
    )
    failures = config.check(results)

    if failures:
        print("EVALUATION GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("EVALUATION GATE PASSED")
        print(f"Results written to: {args.output}")