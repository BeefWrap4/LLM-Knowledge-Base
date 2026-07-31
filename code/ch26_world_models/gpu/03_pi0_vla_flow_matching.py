# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.2.1 Pi0 / Pi0.5 — Physical Intelligence VLA
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 03_pi0_vla_flow_matching.py
# expected_runtime: 5-15s (小 MLP 训练 50 步)
# expected_output: flow matching loss 下降曲线
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.2.1
#
# Interview hooks:
#   1. Flow matching 与 DDPM 的核心区别? (ODE vs SDE, 速度场 vs 噪声预测)
#   2. Pi0 为什么用 flow matching 而非 diffusion? (训练更稳定, 推理步数少)
#   3. VLA 模型中 action expert 的输入是什么? (VLM 特征 + robot state)
"""Pi0 VLA (Vision-Language-Action) flow matching 训练 demo.

Pi0 是 Physical Intelligence 的 VLA 模型, 用 flow matching 替代 diffusion:
  - 输入: 图像 + 语言指令
  - 输出: 动作轨迹 (7-DoF robot arm)
  - 训练: 噪声→动作的 flow ODE 学习

本 demo 用合成 (action, cond) 数据 + 小 MLP 演示 flow matching loss.
生产 Pi0: transformer + 真实机器人数据 (Physical Intelligence 内部).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


class SimpleFlowMatchingMLP(nn.Module):
    """小 MLP, 演示 flow matching 用 (生产 Pi0 用 transformer + 视觉编码器)."""

    def __init__(self, action_dim: int = 7, cond_dim: int = 64, hidden: int = 128):
        super().__init__()
        # 输入: action (noisy) + t + cond → 输出: 速度场 v
        self.net = nn.Sequential(
            nn.Linear(action_dim + cond_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, action_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # action_t: [B, action_dim], t: [B, 1], cond: [B, cond_dim]
        x = torch.cat([action_t, t, cond], dim=-1)
        return self.net(x)


def flow_matching_loss(
    model: SimpleFlowMatchingMLP,
    action: torch.Tensor,
    cond: torch.Tensor,
    sigma_min: float = 0.001,
) -> torch.Tensor:
    """Flow matching loss: 预测速度场 v = a - x_0.

    算法 (Lipman et al. 2023):
      1. 采样 t ~ U(0, 1), 噪声 ε ~ N(0, I)
      2. 构造插值: x_t = (1 - (1-σ_min) * t) * ε + t * action
      3. 目标速度场: v_target = action - (1-σ_min) * ε
      4. loss = MSE(model(x_t, t, cond), v_target)
    """
    B = action.size(0)
    t = torch.rand(B, 1, device=action.device)
    noise = torch.randn_like(action)
    x_t = (1.0 - (1.0 - sigma_min) * t) * noise + t * action
    target_v = action - (1.0 - sigma_min) * noise
    pred_v = model(x_t, t, cond)
    return ((pred_v - target_v) ** 2).mean()


def main() -> None:
    check_hardware()
    print("=== Pi0 VLA Flow Matching 训练 demo ===\n")
    print("核心: 用 ODE 速度场 v = da/dt 替代 DDPM 噪声预测")
    print("     训练: 噪声 → 动作的直线 ODE 路径 (vs diffusion 的随机游走)")
    print()

    # 合成数据: 32 样本, action=7-DoF (末端执行器), cond=64-D (VLM 特征)
    B = 32
    action = torch.randn(B, 7).cuda()
    cond = torch.randn(B, 64).cuda()

    model = SimpleFlowMatchingMLP(action_dim=7, cond_dim=64).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型: 7→128→128→7 MLP, 参数量 {n_params:,}")
    print(f"  训练: 50 步, batch={B}\n")

    losses = []
    for step in range(50):
        loss = flow_matching_loss(model, action, cond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"  step {step:3d} | flow_matching_loss = {loss.item():.4f}")

    print(f"\n  ✅ loss 下降: {losses[0]:.4f} → {losses[-1]:.4f}")

    # 推理: 从噪声 ODE 积分 10 步 (Euler) → 动作
    print("\n  推理 demo: 噪声 → ODE 积分 → 动作")
    noise = torch.randn(1, 7).cuda()
    cond_sample = cond[:1]
    dt = 1.0 / 10
    x = noise
    print(f"    起点 (噪声): {x[0, :3].tolist()}")
    for k in range(10):
        t = torch.full((1, 1), k * dt).cuda()
        v = model(x, t, cond_sample)
        x = x + v * dt
    print(f"    终点 (动作): {x[0, :3].tolist()}")
    print(f"    真实目标  : {action[0, :3].tolist()}")

    print()
    print("=" * 60)
    print("π0 论文/模型卡背景（不是本脚本的运行证据）:")
    print("  - 开放 checkpoint 的 VLM backbone 与 action expert 以当前模型卡为准")
    print("  - flow-matching 积分步数、动作 chunk 和控制频率是部署配置，不作通用承诺")
    print("  - 本脚本只拟合合成张量，未加载 lerobot/pi0_base、机器人数据或控制器")


if __name__ == "__main__":
    main()
    print("OK")
