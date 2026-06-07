# ---
# chapter: 27
# topic: GRPO loss deep-dive (no-critic group-relative)
# section: 27.6.2 RL 阶段
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: torch>=2.0 (optional) ; 纯 numpy 也可跑
# run: python 05_grpo_loss.py
# expected_runtime: <3s
# expected_output: 打印 GRPO loss 与优势数值随 step 下降
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.6.2 + §27.8 Q3
# Interview hooks:
#   1. GRPO 与 PPO 核心区别？省掉了什么网络？
#   2. KL 散度在 GRPO loss 中位置（直接加 vs 单独奖励）？
#   3. 组大小 G 对优势估计方差的影响？
"""GRPO (Group Relative Policy Optimization) 损失实现。

DeepSeek-R1 / R1-Zero 使用的去 Critic 算法。对同一 prompt 采样 G 个回答，
组内归一化得到优势，无需 value network。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GRPOConfig:
    group_size: int = 4       # G
    kl_coef: float = 0.04     # β
    clip_ratio: float = 0.2   # ε，PPO 风格
    lr: float = 1e-5


def group_advantages(rewards: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """A_i = (r_i - mean(r)) / (std(r) + eps)"""
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


def grpo_loss(
    log_probs_new: np.ndarray,   # 新策略下 G 个回答的 log π(a|q)
    log_probs_old: np.ndarray,   # 旧策略下 log π_old
    log_probs_ref: np.ndarray,   # 参考策略 (SFT 模型) 下 log π_ref
    rewards: np.ndarray,         # G 个标量奖励
    cfg: GRPOConfig,
) -> tuple[float, np.ndarray]:
    """GRPO 单步目标 (简化版，去掉了 token-level 分解)。

    L = -E[ min(ratio·A, clip(ratio, 1-ε, 1+ε)·A) ] + β · KL(π || π_ref)
    ratio = exp(log π_new - log π_old)
    """
    advantages = group_advantages(rewards)
    ratio = np.exp(log_probs_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * advantages
    policy_loss = -np.minimum(surr1, surr2).mean()

    # KL 近似：k3 estimator (Schulman 2020)
    diff = log_probs_ref - log_probs_new  # log(π_ref/π_new)
    kl = (np.exp(diff) - diff - 1.0).mean()
    total = policy_loss + cfg.kl_coef * kl

    return float(total), advantages


def main() -> None:
    rng = np.random.default_rng(0)
    cfg = GRPOConfig()
    print(f"GRPO config: G={cfg.group_size}, β={cfg.kl_coef}, "
          f"ε={cfg.clip_ratio}\n")

    # 模拟 20 步训练：策略从随机 → 高奖励回答
    G = cfg.group_size
    for step in range(20):
        rewards = rng.uniform(0, 1, size=G)
        # 让"idx 0"在训练中逐渐成为最高奖励
        rewards[0] += step * 0.05
        log_new = rng.normal(0, 1, G)
        log_old = log_new + rng.normal(0, 0.1, G)  # 旧策略略不同
        log_ref = np.zeros(G)  # 参考策略 log prob
        loss, adv = grpo_loss(log_new, log_old, log_ref, rewards, cfg)
        print(
            f"step {step:>2}  loss={loss:+.3f}  "
            f"rewards={rewards.round(2)}  adv={adv.round(2)}"
        )

    print("\n关键观察：")
    print("  • G=4 时组内优势方差较大 → 训练噪声大 → 生产中常取 G=8~16")
    print("  • KL 约束让策略不远离 SFT 模型，避免奖励黑客")


if __name__ == "__main__":
    main()
