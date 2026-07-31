# ---
# chapter: 27
# topic: GRPO advantage estimation and group size effect
# section: 27.6.2 RL 阶段
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: torch>=2.0
# run: python 06_grpo_advantage.py
# expected_runtime: <2s (pure PyTorch)
# expected_output: prints group-relative advantage values per prompt
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.6.2
# Interview hooks:
#   1. 为什么 GRPO 不需要 value network？组内 baseline 优势在哪？
#   2. 组大小 G=1 vs G=16 对 advantage 估计的影响？什么情况下退化为 REINFORCE？
#   3. 同一组内 reward 完全相同时 advantage 的退化情况？
"""GRPO Advantage 计算: 组内标准化.

advantage_i = (r_i - mean(r_group)) / std(r_group)

无 critic model, 用组内 reward 标准化作为 baseline.
"""

import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("[SKIP] 需要 torch>=2.0；请安装 GPU tier 依赖")
    print("OK")
    raise SystemExit(0)

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))


def grpo_advantage(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算 GRPO advantage: 对每个组, 标准化 rewards.

    Args:
        rewards: [B] 每个回答的奖励
        group_ids: [B] 标识 prompt 组 (同一 prompt 的 K 个回答同组)

    Returns:
        [B] advantage 张量
    """
    advantages = torch.zeros_like(rewards)
    for gid in group_ids.unique():
        mask = group_ids == gid
        if mask.sum() < 2:
            # 单回答组, advantage=0
            continue
        group_r = rewards[mask]
        advantages[mask] = (group_r - group_r.mean()) / (group_r.std() + eps)
    return advantages


def main():
    print("=== GRPO Advantage 计算 ===\n")

    # 4 个 prompt, 每个 4 个回答
    rewards = torch.tensor(
        [
            # prompt 0: 4 个回答, rewards 0.9, 0.5, 0.3, 0.1
            0.9,
            0.5,
            0.3,
            0.1,
            # prompt 1: 4 个回答, rewards 0.8, 0.4, 0.6, 0.2
            0.8,
            0.4,
            0.6,
            0.2,
            # prompt 2: 4 个回答, rewards 1.0, 0.0, 0.5, 0.5
            1.0,
            0.0,
            0.5,
            0.5,
            # prompt 3: 4 个回答, rewards 0.7, 0.7, 0.7, 0.7
            0.7,
            0.7,
            0.7,
            0.7,
        ]
    )
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

    advantages = grpo_advantage(rewards, group_ids)

    print(f"  rewards:   {rewards.tolist()}")
    print(f"  group_ids: {group_ids.tolist()}")
    print(f"  advantages: {[f'{a:+.3f}' for a in advantages.tolist()]}")
    print()

    for gid in [0, 1, 2, 3]:
        mask = group_ids == gid
        print(
            f"  prompt {gid}: rewards={[f'{r:.2f}' for r in rewards[mask].tolist()]}, adv={[f'{a:+.3f}' for a in advantages[mask].tolist()]}"
        )

    print("\n  注意: prompt 3 (所有 reward 相同) → advantage 全部 0 (无 baseline)")
    print("  优势: 组内相对排序, 无需学 critic model")
    print("OK")


if __name__ == "__main__":
    main()
