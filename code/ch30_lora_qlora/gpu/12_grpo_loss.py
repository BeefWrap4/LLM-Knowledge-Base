# ---
# chapter: 31
# topic: 偏好对齐与强化学习
# topic_id: lora_qlora.grpo_loss
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 12_grpo_loss.py
# expected_runtime: <3s
# expected_output: GRPO loss 值 + .backward() 验证可微
# ---
# See: ../../../31_偏好对齐与强化学习.md
#
# Interview hooks:
#   1. GRPO 相对 PPO 的核心简化？为什么可以去掉 Critic / Value Network？
#   2. 组内 z-score 归一化的优势？和 GAE 优势估计的本质差异？
#   3. KL 惩罚项 beta 如何调节？beta 太大太小分别有什么后果？
"""GRPO (Group Relative Policy Optimization) 训练 demo.

GRPO 是 DeepSeek-R1 用的 RL 算法:
  - 对每个 prompt 采样 K 个回答
  - 用组内相对 reward (而非绝对) 计算 advantage
  - 简化版 PPO (无 critic model, 用组内 z-score 估计 baseline)

Loss = clip_surrogate + β * KL(policy || ref)
"""

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    """纯 GPU tensor 计算, 1GB 即可; 保留接口统一."""
    require_nvidia_gpu(min_vram_gb=0, min_count=1)


def grpo_loss(
    log_probs: torch.Tensor,  # [B*K, T] 当前策略 token-level logp
    old_log_probs: torch.Tensor,  # [B*K, T] 旧策略 token-level logp
    ref_log_probs: torch.Tensor,  # [B*K, T] 参考策略 token-level logp
    rewards: torch.Tensor,  # [B*K]   每条样本的标量 reward
    beta: float = 0.04,
    clip_range: float = 0.2,
    group_size: int = 1,
) -> torch.Tensor:
    """GRPO loss = clipped surrogate + β * KL penalty.

    L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] + β * KL(π || π_ref)
    """
    if group_size < 2 or rewards.numel() % group_size != 0:
        raise ValueError("group_size 必须 >=2，且 rewards 数量必须能被 group_size 整除")

    # 1) 每 group_size 条共享一个 prompt；只在各 prompt 的候选组内标准化。
    grouped_rewards = rewards.reshape(-1, group_size)
    group_mean = grouped_rewards.mean(dim=1, keepdim=True)
    group_std = grouped_rewards.std(dim=1, keepdim=True, unbiased=False)
    advantages = ((grouped_rewards - group_mean) / (group_std + 1e-8)).reshape(-1)

    # 2) Importance sampling ratio + clip
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages.unsqueeze(-1)
    surr2 = ratio.clamp(1 - clip_range, 1 + clip_range) * advantages.unsqueeze(-1)
    policy_loss = -torch.min(surr1, surr2).mean()

    # 3) KL 惩罚 (k3): r = log(pi_ref / pi_policy)
    log_ratio = ref_log_probs - log_probs
    kl = (log_ratio.exp() - log_ratio - 1.0).mean()

    return policy_loss + beta * kl


def main():
    check_hardware()

    print("=== GRPO Loss (真实 PyTorch) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    torch.manual_seed(42)
    # 模拟 4 个 prompt × 3 个回答 = 12 条
    B, T, K = 4, 16, 3
    total = B * K

    log_probs = (torch.randn(total, T, device="cuda") * 0.1 - 1.0).requires_grad_(True)
    old_log_probs = log_probs.detach() + torch.randn_like(log_probs) * 0.01
    ref_log_probs = torch.randn(total, T, device="cuda") * 0.1 - 1.0

    # 模拟组内 reward (组内高低不齐 → z-score 后 advantages 正负对称)
    rewards = torch.rand(total, device="cuda")
    print(f"  log_probs shape:    {tuple(log_probs.shape)}")
    print(f"  rewards shape:      {tuple(rewards.shape)}")
    print(f"  rewards range:      [{rewards.min().item():.3f}, {rewards.max().item():.3f}]")

    loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, rewards, group_size=K)
    loss.backward()

    print(f"  loss:               {loss.item():.4f}")
    print(f"  loss requires_grad: {log_probs.requires_grad}")
    print(f"  log_probs.grad norm: {log_probs.grad.norm().item():.4f}")
    print("\n  该教学目标函数可微，并正确执行组内 advantage 标准化")
    print("  生产 GRPO 还需 response mask、按 token 聚合、分布式采样与稳定性监控")
    print("OK")


if __name__ == "__main__":
    main()
