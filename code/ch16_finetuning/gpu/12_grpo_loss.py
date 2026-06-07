# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.11.1 GRPO 损失伪代码
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 12_grpo_loss.py
# expected_runtime: <5s
# expected_output: 用 mock 张量演示 GRPO 损失计算过程, 打印 3 个数值
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.11.1
# Interview hooks:
#   1. GRPO 相对 PPO 的核心简化？为什么可以去掉 Critic / Value Network？
#   2. 组内 z-score 归一化的优势？和 GAE 优势估计的本质差异？
#   3. KL 惩罚项 beta 如何调节？beta 太大太小分别有什么后果？



"""
GRPO 损失 PyTorch 风格伪代码

GRPO = Group Relative Policy Optimization
- 对每个 prompt 采样 G 个回答
- 组内 z-score 归一化得到优势
- 重要性采样比 + clip + KL 惩罚
"""

from __future__ import annotations
import torch


def grpo_loss(
    log_probs: torch.Tensor,        # (B, T)  当前策略每个 token 的对数概率
    old_log_probs: torch.Tensor,    # (B, T)  旧策略每个 token 的对数概率
    ref_log_probs: torch.Tensor,    # (B, T)  参考策略每个 token 的对数概率
    rewards: torch.Tensor,          # (B,)    每条样本的标量奖励
    group_ids: torch.Tensor,        # (B,)    同一 prompt 的样本共享同一 group_id
    beta: float = 0.04,
    eps: float = 0.2,
):
    """
    简化版 GRPO 损失 (单步 PPO-style update)
    """
    # ---- 1) 组内 z-score 归一化得到优势 ----
    # 用 scatter 模拟 groupby
    B = rewards.shape[0]
    mean = torch.zeros_like(rewards)
    std = torch.zeros_like(rewards)
    for gid in group_ids.unique():
        mask = group_ids == gid
        m = rewards[mask].mean()
        s = rewards[mask].std().clamp_min(1e-4)
        mean[mask] = m
        std[mask] = s
    advantages = (rewards - mean) / std                  # (B,)

    # ---- 2) 重要性采样比 + clip ----
    ratio = torch.exp(log_probs - old_log_probs)          # (B, T)
    surr1 = ratio * advantages.unsqueeze(-1)
    surr2 = ratio.clamp(1 - eps, 1 + eps) * advantages.unsqueeze(-1)
    policy_loss = -torch.min(surr1, surr2).sum(-1).mean()

    # ---- 3) 与参考策略的 KL 惩罚 (k3 估计器) ----
    # KL ≈ (p / p_ref) - log(p / p_ref) - 1 = exp(log_ratio) - log_ratio - 1
    log_ratio_ref = log_probs - ref_log_probs
    kl = (log_ratio_ref.exp() - log_ratio_ref - 1.0).sum(-1).mean()

    total_loss = policy_loss + beta * kl
    return total_loss, policy_loss.detach(), kl.detach()


if __name__ == "__main__":
    torch.manual_seed(42)

    # 模拟一个 group: 2 个 prompt, 每个 prompt 4 个回答 -> B=8
    B, T = 8, 16
    log_probs = torch.randn(B, T) * 0.5
    old_log_probs = log_probs + torch.randn(B, T) * 0.1
    ref_log_probs = torch.randn(B, T) * 0.5
    # 2 个 group, 每个 4 条
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    # rewards 在 group 内有高低, 模拟正确答案奖励更高
    rewards = torch.tensor([0.8, 0.3, 0.6, 0.1, 0.9, 0.4, 0.7, 0.2])

    total, policy, kl = grpo_loss(log_probs, old_log_probs, ref_log_probs,
                                  rewards, group_ids)
    print(f"[GRPO mock] policy_loss = {policy.item():.4f}")
    print(f"[GRPO mock] kl_penalty  = {kl.item():.4f}")
    print(f"[GRPO mock] total_loss  = {total.item():.4f}  (beta=0.04)")
    print()
    print("OK")