# ---
# chapter: 18
# topic: Context Engineering
# topic_id: context_engineering.context_rot_demo
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 03_context_rot_demo.py
# expected_runtime: <1s
# ---
#
# See: ../../../18_Context_Engineering.md
# Evidence:
#   - Lost in the Middle (TACL 2024): https://aclanthology.org/2024.tacl-1.9/
#   - Chroma Context Rot (2025): https://www.trychroma.com/research/context-rot
#
# Important:
#   本文件不会调用模型，也不会生成实验数据。曲线仅用于解释部分长上下文评测中
#   出现过的 U 型位置偏差；它不代表任何具体模型、上下文长度或准确率。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticPoint:
    """合成教学曲线上的一个点；score 是无量纲示意值，不是召回率。"""

    position_fraction: float
    score: float


def synthetic_position_score(position_fraction: float) -> float:
    """返回对称 U 型示意值。

    公式和参数是人为选择的，只表达“边缘高、中间低”的形状，不拟合任何模型。
    """

    if not 0.0 <= position_fraction <= 1.0:
        raise ValueError("position_fraction 必须在 [0, 1] 内")
    distance_from_middle = abs(2.0 * position_fraction - 1.0)
    return 0.45 + 0.50 * distance_from_middle**1.5


def build_synthetic_curve() -> list[SyntheticPoint]:
    """生成固定位置上的合成曲线，便于离线教学与回归测试。"""

    positions = (0.05, 0.25, 0.50, 0.75, 0.95)
    return [
        SyntheticPoint(position_fraction=position, score=synthetic_position_score(position))
        for position in positions
    ]


def real_benchmark_checklist() -> tuple[str, ...]:
    """真实长上下文 benchmark 的最低设计要求。"""

    return (
        "固定模型 ID/快照、解码参数、系统提示和评测日期",
        "交叉测试多个输入长度与首/中/尾多个证据位置",
        "同时覆盖词面匹配、语义匹配、无干扰/多干扰和连贯/打乱文本",
        "使用足量样本或随机种子，报告准确率、拒答率和置信区间",
        "保存原始输入输出，并记录实际 token、延迟和费用",
    )


def run_demo() -> None:
    print("=== U 型位置偏差：合成教学曲线（不是模型实验） ===")
    print("警告：下列 score 不是准确率，不代表任何具体模型或 token 长度。\n")
    print(f"{'证据位置':>10s} | {'合成 score':>12s} | 示意")
    print("-" * 48)
    for point in build_synthetic_curve():
        bar = "█" * round(point.score * 20)
        print(f"{point.position_fraction:>9.0%} | {point.score:>12.3f} | {bar}")

    print("\n真实 benchmark 最低要求：")
    for index, item in enumerate(real_benchmark_checklist(), start=1):
        print(f"  {index}. {item}")

    print("\n结论：是否存在位置偏差、长度退化及其幅度，必须在目标模型和任务上实测。")


if __name__ == "__main__":
    run_demo()
    print("\nOK")
