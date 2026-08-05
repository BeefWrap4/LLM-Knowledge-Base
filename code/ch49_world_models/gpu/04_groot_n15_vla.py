# ---
# chapter: 49
# topic: 世界模型、VLA 与具身智能
# topic_id: world_models.groot_n15_vla
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 04_groot_n15_vla.py
# expected_runtime: 5-15s (small illustrative MLP, 50 steps)
# expected_output: illustrative action-expert loss decreases + explicit GR00T boundary
# ---
# See: ../../../49_世界模型VLA与具身智能.md
#
# Interview hooks:
#   1. 当前 GR00T N1.7 的公开架构边界是什么？
#   2. 为什么这个 MLP 不能冒充 GR00T 的 diffusion-transformer action head？
#   3. 从开放 checkpoint 到目标机器人部署还缺哪些数据、安全与评估步骤？
"""通用 action-expert 接口教学；不是 GR00T N1.5/N1.7 的实现或复现。

文件名是历史遗留。当前 NVIDIA 官方仓库主线为 GR00T N1.7 Early Access，公开说明包含
Cosmos-Reason2-2B（Qwen3-VL 架构）VLM backbone、diffusion-transformer action head 与
3B base checkpoint。本例只用随机合成张量训练一个 MLP，不能证明官方架构、数据、checkpoint、
机器人任务效果或部署可用性。
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
    """这里只检查小型 CUDA 教学循环；不是官方 GR00T 硬件规格。"""
    require_nvidia_gpu(min_vram_gb=2, min_count=1)


class ActionExpertMLP(nn.Module):
    """教学 Action expert：合成视觉语言特征 + 状态 → 连续动作。

    这里的维度和 MLP 都是示例设置，只演示 gradient flow，不对应 GR00T checkpoint。
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
    print("=== 通用 action-expert 教学（非 GR00T 实现）===\n")
    print("示意: synthetic VLM feature + robot state → MLP → continuous action")
    print()

    B = 8
    # 2048-D 只是教学维度，不对应当前 GR00T VLM hidden size。
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
    print("当前 GR00T 事实边界（以 NVIDIA Isaac-GR00T 官方仓库为准）:")
    print("  - 当前主线: N1.7 Early Access；旧 N1.5/N1.6 是历史版本")
    print("  - 公开 base checkpoint: nvidia/GR00T-N1.7-3B")
    print("  - 官方描述: Cosmos-Reason2-2B backbone + diffusion-transformer action head")
    print("  - 本脚本未加载 checkpoint、LeRobot 数据、Isaac 环境或机器人控制器")
    print("  - loss 下降只证明这个小 MLP 拟合了同一批随机合成目标")


if __name__ == "__main__":
    main()
    print("OK")
