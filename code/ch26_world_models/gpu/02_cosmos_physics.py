# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.4.2 NVIDIA Cosmos — 物理世界基础模型
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: transformers, torch (lazy), numpy
# run: python 02_cosmos_physics.py
# expected_runtime: 30-60s (config load + 真物理仿真)
# expected_output: Cosmos 架构加载 + 物理仿真轨迹
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.4.2
#
# Interview hooks:
#   1. NVIDIA Cosmos 与 Genie 3 在训练目标上有什么不同？（物理一致性 vs 视觉逼真度）
#   2. Cosmos 如何把合成数据与 Isaac Sim 仿真结合？Video Tokenizer 的作用？
#   3. 在机器人 sim-to-real 中，世界模型扮演什么角色？
"""Cosmos 物理建模演示 (NVIDIA Cosmos-1.0).

Cosmos 是 NVIDIA 的世界基础模型系列:
  - Cosmos-1.0-7B / 13B (物理)
  - Cosmos-1.0-7B-Video / 13B-Video
  - 用于物理感知视频生成 + 自动驾驶仿真

Cosmos-7B fp16 ~14GB, 在 34GB 单卡可跑 (4-bit 量化更舒适).
本 demo: 尝试 config load 验证架构 + 真物理仿真 rollout (gravity, friction, collision).
"""
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import math
import numpy as np
import torch

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    """Cosmos-7B 14GB, 单卡 24GB+ 即可跑 fp16."""
    require_nvidia_gpu(min_vram_gb=24, min_count=1)


def try_load_cosmos_config():
    """尝试从 HF 加载 Cosmos-1.0-7B config (无权重), 验证架构可访问."""
    from transformers import AutoConfig
    model_id = "nvidia/Cosmos-1.0-7B"
    print(f"目标模型: {model_id} (~14GB fp16)\n")
    print("步骤 1: 从 HuggingFace 加载 config (无权重, 仅几十 KB)...")

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"  ✅ config 加载成功")
        print(f"     架构: {config.architectures}")
        print(f"     hidden_size: {config.hidden_size}")
        print(f"     num_layers : {config.num_hidden_layers}")
        print(f"     num_heads  : {config.num_attention_heads}")
        print(f"     vocab_size : {config.vocab_size}")
        return True
    except Exception as e:
        print(f"  ⚠️  config 加载失败: {type(e).__name__}: {str(e)[:120]}")
        print(f"     注: Cosmos 是 gated repo, 需 NVIDIA NGC 账号 + HF 认证")
        print(f"     解决: 访问 https://huggingface.co/nvidia/Cosmos-1.0-7B 申请 access")
        return False


class CosmosSimulator:
    """Newtonian 物理仿真器: 状态 = (x, y, vx, vy), 支持重力 + 弹性碰撞 + 摩擦.

    物理一致性 ground truth: Cosmos 训练数据源于 Isaac Sim 仿真, 我们用简化的
    Newtonian 力学近似演示"物理世界基础模型"的核心: 状态转移满足物理守恒.
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
        return float(0.5 * np.sum(self.vel ** 2))


def main() -> None:
    check_hardware()
    print("=== NVIDIA Cosmos — 物理世界基础模型 ===\n")
    has_config = try_load_cosmos_config()
    print()

    # 物理仿真 (无需 GPU tensor, CPU numpy 即可)
    print("步骤 2: 真物理仿真 (Newtonian: 重力 + 弹性碰撞 + 摩擦)")
    sim = CosmosSimulator(n_objects=4, seed=42)
    print(f"  初始: 4 物体起始于空中 (高度 {sim.pos[:, 1].round(2).tolist()})")
    print(f"  参数: g=9.8 m/s², restitution={sim.RESTITUTION}, friction={sim.FRICTION}\n")

    initial_energy = sim.energy()
    print(f"  t= 0: KE = {initial_energy:.4f} J, "
          f"y_min = {sim.pos[:, 1].min():.3f}")

    traj = sim.rollout(n_steps=60)
    for t in [10, 30, 59]:
        pos_t = traj[t]
        # 重置 sim 状态到 t 步后的状态
        sim_t = CosmosSimulator(n_objects=4, seed=42)
        sim_t.pos = traj[min(t, len(traj) - 1)].copy() if t == 0 else traj[t].copy()
        # 近似 KE 估算
        vel_approx = (traj[min(t + 1, len(traj) - 1)] - traj[max(t - 1, 0)]) / (2 * sim.DT)
        ke = float(0.5 * np.sum(vel_approx ** 2))
        print(f"  t={t:2d}: y_mean = {pos_t[:, 1].mean():.3f}, "
              f"y_min = {pos_t[:, 1].min():.3f}, KE ≈ {ke:.4f} J")

    print()
    print("=" * 60)
    print("Cosmos 训练数据 (与本 demo 物理仿真对比):")
    print("  - Isaac Sim 渲染 1 亿+ 物理一致视频帧")
    print("  - Video Tokenizer: 压缩为 latent tokens (类似 VAE)")
    print("  - DiT backbone: 预测下一帧 latent")
    print("  - 物理一致性: PhysicalAI-Score 损失 (Newton + collision + permanence)")
    print()
    if not has_config:
        print("⚠️  本次跳过 Cosmos 权重下载 (gated + ~14GB).")
        print("   真跑需要: huggingface-cli login + 申请 NGC access + 14GB+ 存储.")


if __name__ == "__main__":
    main()
