# ---
# chapter: 16
# topic: GRPO Loss (真实可微 PyTorch 实现)
# section: 16.11.1
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 12_grpo_loss.py
# expected_runtime: <3s
# expected_output: GRPO loss 值 + .backward() 验证可微
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.11.1
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
) -> torch.Tensor:
    """GRPO loss = clipped surrogate + β * KL penalty.

    L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] + β * KL(π || π_ref)
    """
    B = rewards.shape[0]
    # 1) 组内 z-score 归一化 (假设每 K 条共享 prompt, 此处简化为全 batch z-score)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 2) Importance sampling ratio + clip
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages.unsqueeze(-1)
    surr2 = ratio.clamp(1 - clip_range, 1 + clip_range) * advantages.unsqueeze(-1)
    policy_loss = -torch.min(surr1, surr2).mean()

    # 3) KL 惩罚 (k3 估计器)
    log_ratio = log_probs - ref_log_probs
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

    loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, rewards)
    loss.backward()

    print(f"  loss:               {loss.item():.4f}")
    print(f"  loss requires_grad: {log_probs.requires_grad}")
    print(f"  log_probs.grad norm: {log_probs.grad.norm().item():.4f}")
    print("\n  GRPO loss 可微, 可直接用于 RLHF/GRPO 训练")
    print("  DeepSeek-R1 / 通义千问 QwQ 都用此 loss")


if __name__ == "__main__":
    main()
