# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.3.3 HIL-SERL — 人在回路样本高效 RL
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 07_hil_serl.py
# expected_runtime: 5-15s (SAC Q-network 50 步训练)
# expected_output: SAC Q-loss 下降 + HIL-SERL 流程
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.3.3
#
# Interview hooks:
#   1. HIL-SERL 与传统 RL 区别? (人在回路 → 样本效率提升 10x+)
#   2. SAC 算法的核心? (最大熵 RL + 双 Q 网络 + 温度自调节)
#   3. 人类干预如何编码进 replay buffer? (干预时: action = 专家动作 + bonus reward)
"""HIL-SERL (Human-in-the-Loop Sample Efficient RL) 算法 demo.

HIL-SERL (Luo et al. 2024) = SAC + 人类干预 + 奖励 shaping
  - SAC 策略 (off-policy 最大熵 RL)
  - 人在执行中可干预, 把成功轨迹加入 buffer
  - 奖励: env reward + 干预 bonus + human preference

本 demo: 真实 SAC Q-network 训练 loop (50 步), 演示 TD loss.
生产 HIL-SERL: 配合 LeRobot + 真实遥操硬件.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


class QNetwork(nn.Module):
    """SAC 双 Q 网络之一 (生产 SAC 用两个 Q + 延迟更新 target)."""

    def __init__(self, state_dim: int = 14, action_dim: int = 7, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


def sac_q_loss(
    q_net: QNetwork,
    target_q: QNetwork,
    state: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """SAC Q loss: MSE(pred_Q, TD target).

    TD target = r + γ * Q_target(s', a'~π)
    生产: 实际用 policy network 采 a' + 熵正则, 此处简化为随机动作.
    """
    with torch.no_grad():
        # 简化: 实际用 policy network 采样 (此处 random proxy)
        next_action = torch.tanh(torch.randn_like(action))  # 限幅到 [-1, 1]
        next_q = target_q(next_state, next_action)
        target = reward + gamma * next_q
    pred = q_net(state, action)
    return F.mse_loss(pred, target)


def main() -> None:
    check_hardware()
    print("=== HIL-SERL 算法 demo (SAC Q-network 训练) ===\n")
    print("核心: SAC 离线策略 + 人类干预加入 replay buffer + 奖励 shaping")
    print()

    B = 32
    state = torch.randn(B, 14).cuda()
    action = torch.randn(B, 7).cuda()
    reward = torch.randn(B, 1).cuda()
    next_state = torch.randn(B, 14).cuda()

    q_net = QNetwork().cuda()
    target_q = QNetwork().cuda()
    target_q.load_state_dict(q_net.state_dict())  # 初始化 target = q_net
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in q_net.parameters())

    print(f"  Q 网络: 14+7 → 256 → 256 → 1, 参数量 {n_params:,}")
    print("  训练: 50 步 TD 学习, γ=0.99\n")

    losses = []
    for step in range(50):
        loss = sac_q_loss(q_net, target_q, state, action, reward, next_state)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 简化: 每 10 步 sync target (生产用 Polyak averaging 0.005)
        if (step + 1) % 10 == 0:
            target_q.load_state_dict(q_net.state_dict())
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"  step {step:3d} | SAC Q loss = {loss.item():.4f}")

    print(f"\n  ✅ loss 下降: {losses[0]:.4f} → {losses[-1]:.4f}")

    # 演示: 预测 Q 值
    print("\n  推理 demo: 评估 (state, action) 的 Q 值")
    with torch.no_grad():
        q_value = q_net(state[:1], action[:1])
    print(f"    Q(s, a) = {q_value.item():.4f}")

    print()
    print("=" * 60)
    print("HIL-SERL 完整流程 (Luo et al. 2024):")
    print("  1. 专家演示 (含人类干预) → replay buffer")
    print("  2. SAC 策略 + 奖励模型联合训练 (off-policy)")
    print("  3. 在线 fine-tune, 持续加入人类干预样本")
    print("  4. 干预信号编码:")
    print("     - 当人介入: action = 专家动作, reward += +1 (成功 bonus)")
    print("     - 当人未介入: action = 策略动作, reward = env reward")
    print("  5. 硬件: 真实遥操 + 机器人 + VR 头显")
    print("  6. 样本效率: 比纯 SAC 提升 10x+ (1-2 小时 vs 1 天)")


if __name__ == "__main__":
    main()
