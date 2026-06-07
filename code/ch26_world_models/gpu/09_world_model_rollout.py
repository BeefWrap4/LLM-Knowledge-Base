# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.4.3 世界模型驱动的 MPC / Planning
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch, numpy
# run: python 09_world_model_rollout.py
# expected_runtime: 5-15s (dynamics model 训练 + 100 步 rollout)
# expected_output: dynamics loss 下降 + imagined rollout 轨迹
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.4.3
#
# Interview hooks:
#   1. DreamerV3 与 Genie 3 在"潜空间动力学"上的核心差异？
#   2. CEM (Cross-Entropy Method) 为什么适合 action sequence 规划？
#   3. 世界模型 rollout 的 horizon 越深越好吗？compounding error 如何缓解？
"""世界模型 rollout 演示 (在想象中训练策略).

核心: 用世界模型预测未来状态, 在想象中训练策略 (model-based RL)
  1. 真实环境收集数据 D = {(s, a, s', r)}
  2. 训练世界模型 f(s, a) → Δs
  3. 用 f 做想象 rollout: ŝ_0, â_0, ŝ_1, â_1, ... (无真实交互)
  4. 在想象轨迹上训练策略 π(a|s)

本 demo: 真训练 dynamics model (50 步) + 100 步 imagined rollout.
生产: DreamerV3 / IRIS / Cosmos-1 在 latent space 做想象.
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


class DynamicsModel(nn.Module):
    """动力学模型: f(s, a) → Δs (状态增量).

    生产: 在 latent space 预测, encoder/decoder 处理图像观测.
    本 demo: 直接在 state space (14-DoF).
    """

    def __init__(self, state_dim: int = 14, action_dim: int = 7, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


def dynamics_loss(model: DynamicsModel, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    """训练: 预测 Δs = s_{t+1} - s_t."""
    pred = model(s, a)
    target = s_next - s
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def imagine_rollout(
    model: DynamicsModel,
    s0: torch.Tensor,
    policy: nn.Module,
    horizon: int = 100,
) -> torch.Tensor:
    """在想象中 rollout horizon 步, 收集轨迹."""
    s = s0
    traj = [s.cpu().numpy()]
    for _ in range(horizon):
        a = policy(s)
        ds = model(s, a)
        s = s + ds
        traj.append(s.cpu().numpy())
    return np.array(traj).squeeze(1)  # [T+1, state_dim]


class SimplePolicy(nn.Module):
    """随机策略 (生产用 SAC/PPO + imagination gradient)."""

    def __init__(self, state_dim: int = 14, action_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s) * 0.1  # 小动作


def main() -> None:
    check_hardware()
    print("=== 世界模型 Rollout 演示 (Dreamer-style imagination) ===\n")
    print("核心: 训练 dynamics model + 在想象中 rollout (无真实交互)")
    print()

    # 1. 训练动力学模型
    print("步骤 1: 训练 dynamics model (50 步 MSE)")
    B = 64
    state = torch.randn(B, 14).cuda()
    action = torch.randn(B, 7).cuda() * 0.1
    next_state = state + torch.randn(B, 14).cuda() * 0.05  # 简化: 小随机扰动

    model = DynamicsModel().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())

    losses = []
    for step in range(50):
        loss = dynamics_loss(model, state, action, next_state)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"  step {step:3d} | dynamics loss = {loss.item():.6f}")
    print(f"  ✅ 参数量 {n_params:,}, loss 下降: {losses[0]:.6f} → {losses[-1]:.6f}\n")

    # 2. 想象 rollout 100 步
    print("步骤 2: 想象 rollout 100 步 (无真实交互)")
    policy = SimplePolicy().cuda()
    s0 = torch.randn(1, 14).cuda()
    traj = imagine_rollout(model, s0, policy, horizon=100)
    print(f"  ✅ rollout 完成, 轨迹 shape: {traj.shape} (T+1, state_dim)")
    print(f"  起点 |s|={np.linalg.norm(traj[0]):.3f}")
    print(f"  终点 |s|={np.linalg.norm(traj[-1]):.3f}")
    print(f"  累积位移 = {np.linalg.norm(traj[-1] - traj[0]):.3f}")

    # 3. Compounding error 分析
    print("\n步骤 3: Compounding error 分析 (horizon 越深误差越大)")
    real_states = [s0.cpu().numpy()]
    s = s0
    for t in range(100):
        a = policy(s)
        # "真实" 转移: 与训练数据同分布
        ds_real = torch.randn(1, 14).cuda() * 0.05
        s = s + ds_real
        real_states.append(s.cpu().numpy())
    real_traj = np.array(real_states).squeeze(1)
    # 对比 imagined vs "真实"
    err_per_step = np.linalg.norm(traj - real_traj, axis=1)
    print(f"  误差 (L2) 每 20 步: " + ", ".join(
        f"t={t}→{err_per_step[t]:.3f}" for t in [0, 20, 40, 60, 80, 100]
    ))

    print()
    print("=" * 60)
    print("生产世界模型 (Cosmos / Genie / DreamerV3):")
    print("  - Transformer-based 动力学预测 (latent space)")
    print("  - 联合训练 latent dynamics + reward model")
    print("  - 在想象中训练 SAC/PPO 策略 (节省真实交互 100x+)")
    print("  - 应用: 机器人 sim-to-real, 自动驾驶, 游戏 AI")


if __name__ == "__main__":
    main()
