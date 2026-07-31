# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.4.3 世界模型驱动的 MPC / Planning
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch, numpy
# run: python 09_world_model_rollout.py
# expected_runtime: 5-15s (synthetic dynamics model + 100-step rollout)
# expected_output: train/validation loss + predicted/oracle rollout error
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.4.3
#
# Interview hooks:
#   1. DreamerV3 与 Genie 3 在"潜空间动力学"上的核心差异？
#   2. CEM (Cross-Entropy Method) 为什么适合 action sequence 规划？
#   3. 世界模型 rollout 的 horizon 越深越好吗？compounding error 如何缓解？
"""合成动力学模型 rollout 教学；不是 Genie、Cosmos 或 Dreamer 的复现。

核心: 用世界模型预测未来状态, 在想象中训练策略 (model-based RL)
  1. 真实环境收集数据 D = {(s, a, s', r)}
  2. 训练世界模型 f(s, a) → Δs
  3. 用 f 做想象 rollout: ŝ_0, â_0, ŝ_1, â_1, ... (无真实交互)
  4. 可在通过验证的想象轨迹上辅助规划或策略学习

本例在已知合成动力学上训练一个小 MLP，再用相同动作序列比较预测轨迹与 oracle 轨迹。
它只能说明接口、验证集误差与 horizon 误差的计算方式，不能证明真实环境数据效率或任务收益。
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    """这里只检查小型 CUDA 教学循环，不代表任何官方世界模型门槛。"""
    require_nvidia_gpu(min_vram_gb=2, min_count=1)


class DynamicsModel(nn.Module):
    """动力学模型: f(s, a) → Δs (状态增量).

    生产: 在 latent space 预测, encoder/decoder 处理图像观测.
    本 demo: 直接在 state space (14-DoF).
    """

    def __init__(self, state_dim: int = 14, action_dim: int = 7, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


def dynamics_loss(
    model: DynamicsModel, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor
) -> torch.Tensor:
    """训练: 预测 Δs = s_{t+1} - s_t."""
    pred = model(s, a)
    target = s_next - s
    return ((pred - target) ** 2).mean()


def oracle_transition(
    state: torch.Tensor,
    action: torch.Tensor,
    action_matrix: torch.Tensor,
) -> torch.Tensor:
    """可复现的合成真实动力学，用于生成训练集和 oracle rollout。"""
    action_effect = 0.15 * torch.tanh(action @ action_matrix)
    return 0.98 * state + action_effect + 0.01 * torch.sin(state)


@torch.no_grad()
def imagine_rollout(
    model: DynamicsModel,
    s0: torch.Tensor,
    action_sequence: torch.Tensor,
) -> torch.Tensor:
    """对固定动作序列执行模型 rollout，避免混入不同策略动作。"""
    s = s0
    traj = [s.cpu().numpy()]
    for a in action_sequence:
        ds = model(s, a)
        s = s + ds
        traj.append(s.cpu().numpy())
    return np.array(traj).squeeze(1)  # [T+1, state_dim]


class SimplePolicy(nn.Module):
    """仅用于生成确定性测试动作的未训练小网络。"""

    def __init__(self, state_dim: int = 14, action_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s) * 0.1  # 小动作


def main() -> None:
    check_hardware()
    torch.manual_seed(42)
    print("=== 合成动力学模型 Rollout 教学 ===\n")
    print("边界: 小型 MLP + 已知 oracle；不是官方世界模型、真实机器人或 RL 训练")
    print()

    # 1. 用已知合成动力学构造训练/验证数据。
    print("步骤 1: 在合成 transition 上训练 dynamics model")
    B = 512
    action_matrix = torch.randn(7, 14, device="cuda") / (7**0.5)
    state = torch.randn(B, 14).cuda()
    action = torch.randn(B, 7).cuda() * 0.2
    next_state = oracle_transition(state, action, action_matrix)
    val_state = torch.randn(128, 14).cuda()
    val_action = torch.randn(128, 7).cuda() * 0.2
    val_next_state = oracle_transition(val_state, val_action, action_matrix)

    model = DynamicsModel().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())

    losses = []
    for step in range(100):
        loss = dynamics_loss(model, state, action, next_state)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"  step {step:3d} | dynamics loss = {loss.item():.6f}")
    with torch.no_grad():
        val_loss = dynamics_loss(model, val_state, val_action, val_next_state).item()
    print(f"  参数量 {n_params:,}, train loss: {losses[0]:.6f} → {losses[-1]:.6f}")
    print(f"  held-out validation loss: {val_loss:.6f}\n")

    # 2. 为预测和 oracle 固定同一动作序列。
    print("步骤 2: 对同一动作序列比较 100 步 predicted/oracle rollout")
    policy = SimplePolicy().cuda()
    policy.eval()
    s0 = torch.randn(1, 14).cuda()
    oracle_states = [s0.cpu().numpy()]
    actions = []
    oracle_state = s0
    with torch.no_grad():
        for _ in range(100):
            next_action = policy(oracle_state)
            actions.append(next_action)
            oracle_state = oracle_transition(oracle_state, next_action, action_matrix)
            oracle_states.append(oracle_state.cpu().numpy())

    action_sequence = torch.stack(actions)
    predicted_traj = imagine_rollout(model, s0, action_sequence)
    oracle_traj = np.array(oracle_states).squeeze(1)
    err_per_step = np.linalg.norm(predicted_traj - oracle_traj, axis=1)
    print(f"  predicted shape: {predicted_traj.shape}; oracle shape: {oracle_traj.shape}")
    print(
        "  horizon L2 error: "
        + ", ".join(f"t={t}→{err_per_step[t]:.3f}" for t in [0, 20, 40, 60, 80, 100])
    )

    print()
    print("=" * 60)
    print("解释边界:")
    print("  - 单步 validation loss 与长 horizon error 必须同时报告")
    print("  - 真实项目还需校准、不确定性、分布外/闭环评估和安全约束")
    print("  - 是否降低真实交互成本只能由目标任务对照实验回答")


if __name__ == "__main__":
    main()
    print("OK")
