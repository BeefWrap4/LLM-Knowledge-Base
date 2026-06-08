# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.2.2 GR00T N1.5 — NVIDIA 通用机器人基础模型
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 04_groot_n15_vla.py
# expected_runtime: 5-15s (action expert MLP 训练 50 步)
# expected_output: action expert loss 下降 + Groot N1.5 部署信息
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.2.2
#
# Interview hooks:
#   1. GR00T N1.5 与 Pi0 的架构区别? (Cosmos-Reasoning VLM vs PaliGemma)
#   2. 为什么 VLA 需要单独训 action expert 而非 end-to-end?
#   3. Sim-to-real: Isaac Lab 仿真数据如何与真实数据混合训练?
"""NVIDIA GR00T N1.5 VLA 演示 (action expert 训练 loop).

GR00T N1.5:
  - 基础模型: VLM (Cosmos-Reasoning 70B)
  - 适配: action expert (本 demo 重点)
  - 数据: 真实 + Isaac Lab 仿真混合
  - 训练: 两阶段 (VLM 预训练 + 动作 fine-tune)

本 demo: 训练 action expert (VLM 特征 + state → action) with 合成数据.
生产 GR00T: Isaac Lab 仿真 + 真实遥操数据混合训练.
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
    """GR00T 完整 VLM 需 24GB+ (70B 量化). action expert 训练 8GB+ 即可."""
    require_nvidia_gpu(min_vram_gb=24, min_count=1)


class ActionExpertMLP(nn.Module):
    """Action expert: VLM 特征 + 机器人状态 → 末端执行器动作.

    生产 GR00T: 300M transformer, 输入是 VLM hidden states + 触觉/力矩 token.
    本 demo: 简化为 MLP (演示 gradient flow).
    """

    def __init__(self, vlm_dim: int = 2048, state_dim: int = 14, action_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vlm_dim + state_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, vlm_feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([vlm_feat, state], dim=-1))


def main() -> None:
    check_hardware()
    print("=== NVIDIA GR00T N1.5 VLA (action expert demo) ===\n")
    print("核心: VLM (Cosmos-Reasoning 70B) 输出特征 + robot state → action expert → 7-DoF")
    print()

    B = 8
    # 模拟 VLM 输出: 2048-D 是 Cosmos-Reasoning hidden size
    vlm_feat = torch.randn(B, 2048).cuda()
    # 14-DoF 双臂状态 (7 joint/arm × 2 arms)
    state = torch.randn(B, 14).cuda()
    # 7-DoF 末端执行器动作 (相对位移)
    target_action = torch.randn(B, 7).cuda()

    model = ActionExpertMLP().cuda()
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"  模型: VLM-2048 + State-14 → 1024 → 512 → 7 (参数量 {n_params:,})")
    print("  训练: 50 步 MSE 监督学习\n")

    losses = []
    for step in range(50):
        pred = model(vlm_feat, state)
        loss = ((pred - target_action) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"  step {step:3d} | MSE = {loss.item():.4f}")

    print(f"\n  ✅ loss 下降: {losses[0]:.4f} → {losses[-1]:.4f}")

    # 推理: 单样本 forward
    print("\n  推理 demo:")
    with torch.no_grad():
        pred_action = model(vlm_feat[:1], state[:1])
    print(f"    预测 action: {pred_action[0, :3].tolist()}")
    print(f"    目标 action: {target_action[0, :3].tolist()}")

    print()
    print("=" * 60)
    print("GR00T N1.5 真实部署 (NVIDIA 2025):")
    print("  - VLM 基础: Cosmos-Reasoning 7B/70B")
    print("  - Action expert: 300M transformer")
    print("  - 训练数据: Isaac Lab 仿真 + 真实遥操混合")
    print("  - 部署平台: NVIDIA Jetson Orin (边缘) / HGX H100 (云端)")
    print("  - 硬件栈: Cosmos Tokenizer + Triton 推理")


if __name__ == "__main__":
    main()
