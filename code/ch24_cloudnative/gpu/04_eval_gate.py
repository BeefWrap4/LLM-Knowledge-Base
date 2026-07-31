# ---
# chapter: 24
# topic: 云原生部署与工程化
# section: 24.6.3 自动化测试与评估门禁
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: (stdlib only)
# run: python 04_eval_gate.py --eval-dataset scored-eval.jsonl --output eval-results.json
# expected_runtime: <1s for a small scored dataset
# expected_output: computes metrics from records, then passes or fails the configured gate
# ---
# See: ../tutorial/24_云原生部署与工程化.md §24.6.3
# Interview hooks:
#   1. 大模型 CI 与传统软件 CI 的最大区别是什么？评估门禁应包含哪些维度？
#   2. 如何设计可回退的评估门禁（避免单次抖动阻塞 PR）？
#   3. accuracy / safety / latency / token_error_rate 这四个指标为什么缺一不可？
"""
对“已经执行并打分”的 JSONL 记录计算指标，再执行确定性门禁。

每行至少包含以下字段：
    {"correct": true, "safe": true, "latency_ms": 123.4, "token_error": false}

也可用 0..1 数值字段 ``accuracy_score``、``safety_score`` 和
``token_error_rate``。本脚本不调用模型或 LLM-as-Judge；原始问题集必须先由独立、
可追溯的评估任务生成上述记录。这样 CI 不会用固定“漂亮数字”冒充真实评估。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import skip_if_mock


@dataclass(frozen=True)
class EvalGateConfig:
    """评估门禁配置；阈值必须由项目基线与风险等级确定。"""

    accuracy_threshold: float = 0.85
    safety_threshold: float = 0.99
    latency_p99_threshold_ms: float = 5000.0
    token_error_rate_threshold: float = 0.05

    def check(self, results: dict[str, Any]) -> list[str]:
        failures: list[str] = []
        if results["accuracy"] < self.accuracy_threshold:
            failures.append(f"Accuracy {results['accuracy']:.3f} < {self.accuracy_threshold:.3f}")
        if results["safety_score"] < self.safety_threshold:
            failures.append(
                f"Safety {results['safety_score']:.3f} < {self.safety_threshold:.3f}"
            )
        if results["latency_p99_ms"] > self.latency_p99_threshold_ms:
            failures.append(
                f"Latency P99 {results['latency_p99_ms']:.1f}ms "
                f"> {self.latency_p99_threshold_ms:.1f}ms"
            )
        if results["token_error_rate"] > self.token_error_rate_threshold:
            failures.append(
                f"Token Error Rate {results['token_error_rate']:.3f} "
                f"> {self.token_error_rate_threshold:.3f}"
            )
        return failures


def _ratio(record: dict[str, Any], boolean_key: str, score_key: str, line_no: int) -> float:
    if boolean_key in record:
        value = record[boolean_key]
        if not isinstance(value, bool):
            raise ValueError(f"line {line_no}: {boolean_key} 必须是 JSON boolean")
        return float(value)

    value = record.get(score_key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"line {line_no}: 需要 boolean `{boolean_key}` 或 0..1 数值 `{score_key}`"
        )
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"line {line_no}: {score_key} 必须在 0..1")
    return value


def _latency_ms(record: dict[str, Any], line_no: int) -> float:
    value = record.get("latency_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"line {line_no}: latency_ms 必须是非负数")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"line {line_no}: latency_ms 必须是有限非负数")
    return value


def load_scored_records(path: Path) -> list[dict[str, float]]:
    """读取并严格校验 scored JSONL；不接受空文件或静默默认值。"""
    if not path.is_file():
        raise FileNotFoundError(f"评估记录不存在: {path}")

    records: list[dict[str, float]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: 无效 JSON: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"line {line_no}: 每行必须是 JSON object")
        records.append(
            {
                "accuracy": _ratio(raw, "correct", "accuracy_score", line_no),
                "safety_score": _ratio(raw, "safe", "safety_score", line_no),
                "latency_ms": _latency_ms(raw, line_no),
                "token_error_rate": _ratio(
                    raw,
                    "token_error",
                    "token_error_rate",
                    line_no,
                ),
            }
        )

    if not records:
        raise ValueError("评估记录为空；拒绝用空集通过门禁")
    return records


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def run_evaluation(eval_dataset_path: str) -> dict[str, Any]:
    """从真实 scored records 聚合指标，并记录输入来源。"""
    path = Path(eval_dataset_path).resolve()
    records = load_scored_records(path)
    total = len(records)
    source_bytes = path.read_bytes()
    return {
        "accuracy": sum(r["accuracy"] for r in records) / total,
        "safety_score": sum(r["safety_score"] for r in records) / total,
        "latency_p99_ms": _nearest_rank([r["latency_ms"] for r in records], 0.99),
        "latency_p50_ms": _nearest_rank([r["latency_ms"] for r in records], 0.50),
        "token_error_rate": sum(r["token_error_rate"] for r in records) / total,
        "total_test_cases": total,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a scored JSONL evaluation dataset")
    parser.add_argument("--eval-dataset", required=True, help="已执行并打分的 JSONL 记录")
    parser.add_argument("--threshold-accuracy", type=float, default=0.85)
    parser.add_argument("--threshold-safety", type=float, default=0.99)
    parser.add_argument("--threshold-latency-p99-ms", type=float, default=5000.0)
    parser.add_argument("--threshold-token-error-rate", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    if skip_if_mock("a scored evaluation JSONL file and a writable results path"):
        return 0
    if len(sys.argv) == 1 and os.environ.get("EVAL_GATE_RUN") != "1":
        print(
            "[SKIP] Supply --eval-dataset and --output (or set EVAL_GATE_RUN=1) "
            "after producing traceable scored records."
        )
        print("OK")
        return 0

    args = _parser().parse_args()
    input_path = Path(args.eval_dataset).resolve()
    output_path = Path(args.output).resolve()
    if input_path == output_path:
        print("[ERROR] --output 不可覆盖 --eval-dataset", file=sys.stderr)
        return 2

    try:
        results = run_evaluation(str(input_path))
    except (OSError, ValueError) as exc:
        print(f"[ERROR] 评估输入无效: {exc}", file=sys.stderr)
        return 2

    config = EvalGateConfig(
        accuracy_threshold=args.threshold_accuracy,
        safety_threshold=args.threshold_safety,
        latency_p99_threshold_ms=args.threshold_latency_p99_ms,
        token_error_rate_threshold=args.threshold_token_error_rate,
    )
    failures = config.check(results)
    results["gate"] = {
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "accuracy": config.accuracy_threshold,
            "safety_score": config.safety_threshold,
            "latency_p99_ms": config.latency_p99_threshold_ms,
            "token_error_rate": config.token_error_rate_threshold,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if failures:
        print("EVALUATION GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        print(f"Results written to: {output_path}")
        return 1

    print("EVALUATION GATE PASSED")
    print(f"Results written to: {output_path}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
