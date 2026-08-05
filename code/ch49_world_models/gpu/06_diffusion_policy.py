# ---
# chapter: 49
# topic: 世界模型、VLA 与具身智能
# topic_id: world_models.diffusion_policy
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 06_diffusion_policy.py
# expected_runtime: 5-15s (1D UNet 50 步 DDPM 训练)
# expected_output: DDPM 训练 loss 下降 + DDIM 推理轨迹
# ---
# See: ../../../49_世界模型VLA与具身智能.md
#
# Interview hooks:
#   1. Diffusion Policy 为什么能处理多模态动作分布（vs Gaussian/L2 单峰）？
#   2. 与 ACT 的对比：DDPM 去噪过程如何解释为"迭代精化"？
#   3. 推理时使用 DDIM 多少步合适？步数与精度的权衡？
"""Diffusion Policy 训练 (DDPM, 小 1D UNet).

Diffusion Policy (Chi et al., RSS 2023) 用 DDPM 预测动作:
  - 输入: 状态 (state) + 时间步 (t)
  - 输出: 噪声预测
  - 训练: MSE(ε_pred, ε_true)
  - 推理: DDPM/DDIM 50 步去噪 → 动作序列

本 demo: 真实 DDPM 训练 (50 步) + DDIM 推理 (10 步) on 小 1D UNet.
生产 Diffusion Policy: Conv1D UNet 处理时序 + DDIM sampler 加速推理.
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


class SimpleUNet1D(nn.Module):
    """1D UNet for action sequence (简化版 — 生产用 Conv1D U-Net)."""

    def __init__(self, in_dim: int = 7, cond_dim: int = 14, time_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_emb = nn.Linear(cond_dim, time_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim + 2 * time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        c_emb = self.cond_emb(cond)
        return self.net(torch.cat([x_t, t_emb, c_emb], dim=-1))


def linear_beta_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


def make_ddpm_constants(T: int = 100, device: str = "cuda"):
    betas = linear_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars, "T": T}


def ddpm_loss(
    model: SimpleUNet1D,
    x_0: torch.Tensor,
    cond: torch.Tensor,
    consts: dict,
) -> torch.Tensor:
    """DDPM 训练 loss: 预测噪声 ε.

    算法:
      1. 随机采样 t ∈ [0, T)
      2. 采样噪声 ε ~ N(0, I)
      3. 构造 x_t = √α̅_t * x_0 + √(1-α̅_t) * ε
      4. 训练 MSE(ε_pred, ε_true)
    """
    B = x_0.size(0)
    T = consts["T"]
    ab = consts["alpha_bars"]
    t = torch.randint(0, T, (B,), device=x_0.device)
    eps = torch.randn_like(x_0)
    ab_t = ab[t].unsqueeze(-1)  # [B, 1]
    x_t = torch.sqrt(ab_t) * x_0 + torch.sqrt(1.0 - ab_t) * eps
    t_norm = t.float().unsqueeze(-1) / T  # [B, 1] in [0, 1)
    pred_eps = model(x_t, t_norm, cond)
    return ((pred_eps - eps) ** 2).mean()


@torch.no_grad()
def ddim_sample(
    model: SimpleUNet1D,
    cond: torch.Tensor,
    consts: dict,
    n_steps: int = 10,
    in_dim: int = 7,
) -> torch.Tensor:
    """DDIM 推理: 从 N(0, I) 出发 n_steps 去噪 → 干净动作."""
    device = cond.device
    T = consts["T"]
    ab = consts["alpha_bars"]
    # DDIM 时间步 (均匀子集)
    timesteps = torch.linspace(0, T - 1, n_steps + 1, dtype=torch.long, device=device).flip(0)
    x = torch.randn(1, in_dim, device=device)
    for i, t in enumerate(timesteps[:-1]):
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) - 1 else timesteps[-1]
        ab_t = ab[t]
        ab_prev = ab[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
        t_norm = (t.float() / T).unsqueeze(0).unsqueeze(-1)
        pred_eps = model(x, t_norm, cond)
        # 预测 x_0
        x0_hat = (x - torch.sqrt(1.0 - ab_t) * pred_eps) / (torch.sqrt(ab_t) + 1e-8)
        # DDIM 更新 (deterministic)
        x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * pred_eps
    return x


def main() -> None:
    check_hardware()
    print("=== Diffusion Policy DDPM 训练 + DDIM 推理 ===\n")
    print("核心: 用 DDPM 学习 (state → action) 的条件去噪, 推理用 DDIM 加速")
    print()

    B = 32
    action = torch.randn(B, 7).cuda()  # 7-DoF 末端动作
    state = torch.randn(B, 14).cuda()  # 14-DoF 双臂状态

    model = SimpleUNet1D(in_dim=7, cond_dim=14).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    consts = make_ddpm_constants(T=100)

    print(f"  模型: 1D UNet (in=7+time+cond) → 128 → 128 → 7, 参数量 {n_params:,}")
    print("  训练: 50 步 DDPM (T=100)\n")

    losses = []
    for step in range(50):
        loss = ddpm_loss(model, action, state, consts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"  step {step:3d} | DDPM loss = {loss.item():.4f}")

    print(f"\n  ✅ loss 下降: {losses[0]:.4f} → {losses[-1]:.4f}")

    # DDIM 推理
    print("\n  DDIM 10 步推理 (从噪声 → 干净动作):")
    cond_single = state[:1]
    sample = ddim_sample(model, cond_single, consts, n_steps=10, in_dim=7)
    print(f"    推理输出: {sample[0, :3].tolist()}")
    print(f"    目标动作: {action[0, :3].tolist()}")
    print("    (50 步训练未充分, 仅演示流程)")

    print()
    print("=" * 60)
    print("Diffusion Policy 论文路线与工程边界:")
    print("  - 可用时序去噪网络建模动作 chunk")
    print("  - 训练目标、噪声 schedule 与推理步数必须按具体实现/质量延迟实验选择")
    print("  - 多模态优势: 可建模 (state → action) 多峰分布")
    print("  - 本脚本未使用真实视觉观测、机器人数据、LeRobot Trainer 或闭环 rollout")


if __name__ == "__main__":
    main()
    print("OK")
