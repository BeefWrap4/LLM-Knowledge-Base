# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.4.2 NVIDIA Cosmos — 物理世界基础模型
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch (lazy), numpy
# run: MOCK_MODE=1 python 02_cosmos_physics.py
# expected_runtime: <2s
# expected_output: Cosmos 物理一致性度量 + 仿真 rollout 摘要
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.4
# Interview hooks:
#   1. NVIDIA Cosmos 与 Genie 3 在训练目标上有什么不同？（物理一致性 vs 视觉逼真度）
#   2. Cosmos 如何把合成数据与 Isaac Sim 仿真结合？Video Tokenizer 的作用？
#   3. 在机器人 sim-to-real 中，世界模型扮演什么角色？

"""
NVIDIA Cosmos —— 物理一致性世界基础模型（World Foundation Model, WFM）。

本文件演示 Cosmos 的两类核心使用：
  (A) 物理一致性度量（physical-plausibility score）
  (B) 给定初始帧 + 控制信号，rollout 一段仿真轨迹（mock）

真实 Cosmos 模型需 ≥ 80GB GPU；这里用轻量级近似做教学演示。
"""

import os
import math
import numpy as np


MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"


# ---------- A. 物理一致性度量 ----------
def physical_plausibility_score(video: np.ndarray) -> float:
    """简单近似：相邻帧像素差 + 简单重力模型一致性。

    真实 Cosmos 用 PhysicalAI-Score（含 Newtonian 物理损失、
    物体持久性、碰撞守恒等）。这里给出可计算的 proxy。
    """
    if video.ndim != 4:
        raise ValueError("video must be (T, H, W, C)")

    # 1) 帧间差（越小越稳定）
    diff = np.mean(np.abs(video[1:] - video[:-1]))

    # 2) 近似重力检查：若上方像素向下"流动"则加分（极简光流 proxy）
    gray = video.mean(axis=-1)
    downward_flow = (gray[:, 1:, :] - gray[:, :-1, :]).clip(min=0).mean()

    # 3) 物体持久性：相同位置像素不应频繁翻转
    persistence = 1.0 - np.mean(np.abs(np.sign(video[1:] - 0.5) - np.sign(video[:-1] - 0.5)))

    # 合成 0-100 分
    score = 100.0 * (0.5 * math.exp(-10.0 * diff) + 0.3 * persistence + 0.2 * min(1.0, downward_flow * 5))
    return float(round(score, 2))


# ---------- B. 仿真 rollout ----------
class CosmosSimulator:
    """极简 Cosmos 仿真器：状态 = (x, y, vx, vy)，支持重力 + 简单碰撞。"""

    GRAVITY = 9.8
    DT = 1.0 / 30.0  # 30 fps
    GROUND_Y = 0.0

    def __init__(self, n_objects: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.pos = rng.uniform(0.2, 0.8, size=(n_objects, 2))
        self.pos[:, 1] = rng.uniform(0.6, 1.0, size=n_objects)  # start in air
        self.vel = rng.uniform(-0.5, 0.5, size=(n_objects, 2))
        self.vel[:, 1] = rng.uniform(-0.2, 0.2, size=n_objects)

    def step(self, action: np.ndarray) -> np.ndarray:
        """action: (n_objects, 2) 的水平力冲量。"""
        self.vel += action * self.DT
        self.vel[:, 1] -= self.GRAVITY * self.DT
        self.pos += self.vel * self.DT

        # 地面碰撞
        hit = self.pos[:, 1] < self.GROUND_Y
        self.pos[hit, 1] = self.GROUND_Y
        self.vel[hit, 1] = -0.5 * self.vel[hit, 1]  # 0.5 弹性恢复
        self.vel[hit, 0] *= 0.9  # 摩擦
        return self.pos.copy()

    def rollout(self, n_steps: int = 60) -> np.ndarray:
        traj = np.zeros((n_steps, *self.pos.shape))
        for t in range(n_steps):
            action = np.zeros_like(self.pos)
            action[:, 0] = 0.1 * math.sin(t * 0.1)  # 周期性水平推力
            traj[t] = self.step(action)
        return traj


# ---------- main ----------
def main() -> None:
    print("=== NVIDIA Cosmos — Physical AI Demo ===\n")

    # 生成一段 mock 视频：64x64 RGB，30 帧
    rng = np.random.default_rng(7)
    fake_video = rng.uniform(0, 1, size=(30, 64, 64, 3)).astype(np.float32)
    # 制造一点"下落"的视觉模式
    for t in range(30):
        fake_video[t, 32 - t : 32 - t + 4, :, :] *= 0.3  # 暗带往下移
    score = physical_plausibility_score(fake_video)
    print(f"[Cosmos] Physical-plausibility score: {score:.2f} / 100")
    print(f"        (real Cosmos uses PhysicalAI-Score with Newton/collision/permanence losses)")

    sim = CosmosSimulator(n_objects=4, seed=42)
    traj = sim.rollout(n_steps=60)
    final_heights = traj[-1, :, 1]
    print(f"\n[Cosmos] Simulated 60 steps (2s) for 4 objects.")
    print(f"        Final heights: {np.round(final_heights, 3).tolist()}")
    print(f"        Trajectory shape: {traj.shape}  (T, N, xy)")
    print()


if __name__ == "__main__":
    main()
