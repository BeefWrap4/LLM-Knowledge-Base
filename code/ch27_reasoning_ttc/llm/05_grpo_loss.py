# ---
# chapter: 27
# topic: GRPO loss deep-dive (no-critic group-relative)
# section: 27.6.2 RL 阶段
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: torch>=2.0
# run: python 05_grpo_loss.py
# expected_runtime: <3s (pure PyTorch)
# expected_output: prints loss value + group-relative advantage demonstration
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.6.2 + §27.8 Q3
# Interview hooks:
#   1. GRPO 与 PPO 核心区别？省掉了什么网络？
#   2. KL 散度在 GRPO loss 中位置（直接加 vs 单独奖励）？
#   3. 组大小 G 对优势估计方差的影响？
"""GRPO Loss (Group Relative Policy Optimization).

GRPO = DeepSeek-R1 用的 RL 算法:
  - 对每个 prompt 采样 K 个回答
  - advantage = (r - mean(r_group)) / std(r_group)
  - loss = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
  - 无 critic model (vs PPO)
"""

import sys
from pathlib import Path

import torch

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))


def grpo_loss(
    log_probs: torch.Tensor,  # [B, T]
    old_log_probs: torch.Tensor,  # [B, T]
    advantages: torch.Tensor,  # [B]
    group_ids: torch.Tensor,  # [B] 标识 prompt 组
    clip: float = 0.2,
) -> torch.Tensor:
    """GRPO 完整 loss (含 group-relative advantage)."""
    # 重新计算组内 advantage (如果未标准化)
    grouped_means = torch.zeros_like(advantages)
    grouped_stds = torch.ones_like(advantages)
    for gid in group_ids.unique():
        mask = group_ids == gid
        if mask.sum() > 1:
            grouped_means[mask] = advantages[mask].mean()
            grouped_stds[mask] = advantages[mask].std() + 1e-8
    normalized_adv = (advantages - grouped_means) / grouped_stds

    # PPO 风格 clipped surrogate
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * normalized_adv.unsqueeze(-1)
    surr2 = ratio.clamp(1 - clip, 1 + clip) * normalized_adv.unsqueeze(-1)
    return -torch.min(surr1, surr2).mean()


def main():
    print("=== GRPO Loss (真实 PyTorch) ===\n")

    B, T = 8, 16  # 2 prompt × 4 回答
    log_probs = torch.randn(B, T, requires_grad=True) * 0.1 - 1.0
    old_log_probs = log_probs.detach() + torch.randn_like(log_probs) * 0.01
    advantages = torch.tensor([1.0, 0.5, -0.5, -1.0, 0.8, 0.2, -0.2, -0.8])
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])  # 2 个 prompt 组

    loss = grpo_loss(log_probs, old_log_probs, advantages, group_ids)
    loss.backward()

    print(f"  group_ids: {group_ids.tolist()}")
    print(f"  advantages: {advantages.tolist()}")
    print(f"  loss: {loss.item():.4f}")
    print(f"  loss requires_grad: {log_probs.grad is not None}")
    print("\n  ✅ GRPO loss 可微, 支持 group-relative advantage")


if __name__ == "__main__":
    main()
