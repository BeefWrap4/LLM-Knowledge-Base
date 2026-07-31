# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.1 世界模型 — NVIDIA Cosmos 3 元数据边界
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: huggingface_hub (real metadata check), numpy
# run: python 02_cosmos_physics.py
# expected_runtime: depends on network/authentication; toy simulator is local
# expected_output: Cosmos 3 repository metadata check + unrelated Newtonian toy trajectory
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.4.2
#
# Interview hooks:
#   1. NVIDIA Cosmos 3 与 Genie 3 的公开形态和可用性有什么不同？
#   2. 为什么仓库可访问、成功生成视频和物理正确性是三种不同证据？
#   3. 在机器人 sim-to-real 中，世界模型扮演什么角色？
"""Cosmos 3 仓库元数据检查 + 独立 Newtonian 教学模拟器。

截至 2026-07-31，NVIDIA 当前主线是统一的 Cosmos 3 omni-model；早期分立的
Predict/Reason/Transfer 接口不能直接当作当前 SDK 示例。官方仓库列出 Cosmos3-Nano
与 Cosmos3-Super 等 checkpoint。

本例不加载权重、不生成图像/视频，也不验证 Cosmos 的物理一致性或性能。它只尝试读取
Hub 仓库元数据，然后运行一个与 Cosmos 3 完全无关的 NumPy 状态转移示例。
"""

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import math

import numpy as np

from shared.gpu_guard import skip_if_mock


def try_read_cosmos3_metadata() -> bool:
    """读取 Cosmos3-Nano Hub 元数据，不下载权重或推断模型架构。"""
    from huggingface_hub import model_info

    model_id = "nvidia/Cosmos3-Nano"
    print(f"目标仓库: {model_id}\n")
    print("步骤 1: 从 Hugging Face 读取仓库元数据（不下载/加载权重）...")

    try:
        info = model_info(model_id, files_metadata=False)
        print("  ✅ 元数据读取成功")
        print(f"     model_id: {info.id}")
        print(f"     revision: {info.sha or 'not reported'}")
        print(f"     gated: {getattr(info, 'gated', 'not reported')}")
        print(f"     files: {len(info.siblings or [])}")
        return True
    except Exception as e:
        print(f"  ⚠️  元数据读取失败: {type(e).__name__}: {str(e)[:120]}")
        print("     这只说明当前网络/认证/仓库访问未完成，不说明模型不可用。")
        print("     当前模型页: https://huggingface.co/nvidia/Cosmos3-Nano")
        return False


class TeachingPhysicsSimulator:
    """Newtonian 物理仿真器: 状态 = (x, y, vx, vy), 支持重力 + 弹性碰撞 + 摩擦.

    该教学模拟器与 Cosmos 3 的训练数据、架构和评估无关，只用于说明可检查的状态转移。
    """

    GRAVITY = 9.8
    DT = 1.0 / 30.0  # 30 fps
    GROUND_Y = 0.0
    RESTITUTION = 0.5  # 弹性恢复系数
    FRICTION = 0.9  # 地面摩擦衰减

    def __init__(self, n_objects: int = 4, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n = n_objects
        self.pos = rng.uniform(0.2, 0.8, size=(n_objects, 2))
        self.pos[:, 1] = rng.uniform(0.6, 1.0, size=n_objects)  # 起始在空中
        self.vel = rng.uniform(-0.5, 0.5, size=(n_objects, 2))
        self.vel[:, 1] = rng.uniform(-0.2, 0.2, size=n_objects)

    def step(self, action: np.ndarray) -> np.ndarray:
        """action: (n_objects, 2) 的水平力冲量."""
        self.vel += action * self.DT
        self.vel[:, 1] -= self.GRAVITY * self.DT  # 重力加速度
        self.pos += self.vel * self.DT

        # 地面碰撞 + 弹性恢复
        hit = self.pos[:, 1] < self.GROUND_Y
        self.pos[hit, 1] = self.GROUND_Y
        self.vel[hit, 1] = -self.RESTITUTION * self.vel[hit, 1]
        self.vel[hit, 0] *= self.FRICTION
        return self.pos.copy()

    def rollout(self, n_steps: int = 60) -> np.ndarray:
        """生成 T 步轨迹, 应用周期性水平推力."""
        traj = np.zeros((n_steps, *self.pos.shape))
        for t in range(n_steps):
            action = np.zeros_like(self.pos)
            action[:, 0] = 0.1 * math.sin(t * 0.1)
            traj[t] = self.step(action)
        return traj

    def energy(self) -> float:
        """动能 (J), 物理守恒监控指标."""
        return float(0.5 * np.sum(self.vel**2))


def main() -> None:
    if skip_if_mock("network access to read current Cosmos 3 repository metadata"):
        return
    print("=== Cosmos 3 元数据检查 + 独立 Newtonian 教学模拟 ===\n")
    has_metadata = try_read_cosmos3_metadata()
    print()

    # 与 Cosmos 无关的 CPU NumPy 教学模拟。
    print("步骤 2: 独立 Newtonian toy simulator（非 Cosmos 推理）")
    sim = TeachingPhysicsSimulator(n_objects=4, seed=42)
    print(f"  初始: 4 物体起始于空中 (高度 {sim.pos[:, 1].round(2).tolist()})")
    print(f"  参数: g=9.8 m/s², restitution={sim.RESTITUTION}, friction={sim.FRICTION}\n")

    initial_energy = sim.energy()
    print(f"  t= 0: KE = {initial_energy:.4f} J, y_min = {sim.pos[:, 1].min():.3f}")

    traj = sim.rollout(n_steps=60)
    for t in [10, 30, 59]:
        pos_t = traj[t]
        # 重置 sim 状态到 t 步后的状态
        # 近似 KE 估算
        vel_approx = (traj[min(t + 1, len(traj) - 1)] - traj[max(t - 1, 0)]) / (2 * sim.DT)
        ke = float(0.5 * np.sum(vel_approx**2))
        print(
            f"  t={t:2d}: y_mean = {pos_t[:, 1].mean():.3f}, y_min = {pos_t[:, 1].min():.3f}, KE ≈ {ke:.4f} J"
        )

    print()
    print("=" * 60)
    print("边界:")
    print(f"  - metadata_accessible={has_metadata}")
    print("  - 本脚本从未下载或运行 Cosmos 3 权重")
    print("  - 上述轨迹来自独立 NumPy 规则，不是模型生成")
    print("  - 架构、数据与访问步骤以 NVIDIA Cosmos 官方模型卡/仓库为准")


if __name__ == "__main__":
    main()
    print("OK")
