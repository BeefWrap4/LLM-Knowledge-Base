# ---
# chapter: 27
# topic: Process Reward Model (PRM) step-level scoring
# section: 27.4 PRM
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: torch>=2.0
# run: python 09_prm_step_scoring.py
# expected_runtime: <2s (pure PyTorch nn.Module)
# expected_output: prints PRM scores for 5 reasoning steps
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.4
# Interview hooks:
#   1. PRM vs ORM (Outcome Reward Model) 的区别？PRM 数据如何标注？
#   2. PRM 的 Best-of-N vs MCTS 推理时使用方式？
#   3. PRM 在哪些任务上显著优于 ORM？math/code/general？
"""Process Reward Model (PRM) 评分.

PRM 对每一步推理打分, 而非只对最终答案:
  - 训练数据: (state, action, next_state) + step-level reward
  - 推理: 给定 (s_0, a_0, s_1, a_1, ..., s_n), 标每步质量
  - 应用: 推理时搜索 (best-of-N, MCTS)
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path
_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))


class PRMScorer(nn.Module):
    """PRM: 输入 (state, action) 嵌入, 输出 step score."""
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state_emb, action_emb):
        return self.net(torch.cat([state_emb, action_emb], dim=-1))


def main():
    print("=== PRM Step Scoring 演示 ===\n")

    # 5 步推理, 每步 (state, action) 嵌入
    n_steps = 5
    state_emb = torch.randn(n_steps, 64)
    action_emb = torch.randn(n_steps, 64)

    model = PRMScorer(hidden=64)
    scores = model(state_emb, action_emb)

    print(f"  推理步骤数: {n_steps}")
    print(f"  PRM scores: {[f'{s.item():.3f}' for s in scores]}")
    print(f"\n  应用:")
    print(f"    - Best-of-N: 生成 N 个轨迹, 用 PRM 选最高分")
    print(f"    - MCTS: 用 PRM 作 leaf evaluation, 引导 search")
    print(f"    - 训练数据: 自动标注 step reward (vs 人工)")


if __name__ == "__main__":
    main()
