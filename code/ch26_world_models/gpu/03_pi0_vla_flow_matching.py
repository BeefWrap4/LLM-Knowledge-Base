# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.2.1 Pi0 / Pi0.5 — Physical Intelligence VLA
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch (lazy), numpy
# run: MOCK_MODE=1 python 03_pi0_vla_flow_matching.py
# expected_runtime: <3s
# expected_output: Pi0 架构概览 + flow matching 一步 denoise + 动作 token 输出
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.2.1
# Interview hooks:
#   1. Pi0 与传统行为克隆 (BC) 损失的区别？为什么用 flow matching 而不是 L2/MSE？
#   2. Pi0 的动作频率是 50Hz，决策 horizon 是 50 步 —— action chunking 的好处？
#   3. VLA 把"动作"视为 LLM 的特殊 token，这种统一表示的优缺点？

"""
Pi0 (Physical Intelligence) —— 首个生产级 VLA 模型的简化复现。

核心创新：
  - 用 flow matching 训练连续动作分布（不是离散 token）
  - 50Hz 动作输出，chunk size = 50
  - 视觉编码器 (SigLIP) + LLM backbone (3B) + Flow matching head
"""

import os
import math
import numpy as np


MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"


# ---------- 1. Flow Matching 训练目标 ----------
def flow_matching_loss(
    x0: np.ndarray,       # 噪声 (B, T, D)  T=50 步动作 chunk
    x1: np.ndarray,       # 真实动作 (B, T, D)
    pred_v: np.ndarray,   # 模型预测的速度场
) -> float:
    """LFM = E_{t, x0} || v_θ(x_t, t) - (x1 - x0) ||^2

    真实 Pi0 的简化形式：x_t = (1-t)*x0 + t*x1，目标速度 = x1 - x0。
    """
    B = x0.shape[0]
    rng = np.random.default_rng()
    t = rng.uniform(0, 1, size=(B, 1, 1)).astype(np.float32)
    x_t = (1 - t) * x0 + t * x1
    target_v = x1 - x0
    return float(np.mean((pred_v - target_v) ** 2))


# ---------- 2. ODE 推理（Euler / midpoint） ----------
def euler_solve(
    v_net,            # callable(x_t, t) -> velocity
    x0: np.ndarray,   # 初始噪声
    n_steps: int = 10,
) -> np.ndarray:
    """简单 Euler ODE 求解：从 t=0 (noise) 积分到 t=1 (action)。"""
    dt = 1.0 / n_steps
    x = x0
    for k in range(n_steps):
        t_k = k * dt
        v = v_net(x, t_k)
        x = x + dt * v
    return x


class MockVelocityNet:
    """Mock 速度场 —— 实际 Pi0 是预训练 LLM + flow head。"""

    def __init__(self, dim: int = 7, target: np.ndarray = None):
        self.dim = dim
        # 真实模式：target = None
        if target is None:
            target = np.zeros(dim, dtype=np.float32)
        self.target = target

    def __call__(self, x_t: np.ndarray, t: float) -> np.ndarray:
        # 简化：速度 = target - x_t  (随时间线性收敛到 target)
        if x_t.ndim == 1:
            return (self.target - x_t).astype(np.float32)
        return (self.target[None, :] - x_t).astype(np.float32)


# ---------- 3. 完整 Pi0 mini pipeline ----------
class Pi0Mini:
    """极简 Pi0：SigLIP-like encoder + flow head + 50Hz action chunk。"""

    ACTION_DIM = 7         # 6-DOF arm + 1 gripper
    CHUNK_SIZE = 50        # 50 步 ≈ 1s @ 50Hz

    def __init__(self):
        self.v_net = MockVelocityNet(dim=self.ACTION_DIM,
                                     target=np.array([0.1, 0.2, -0.3, 0.0, 0.0, 0.5, 0.8], np.float32))

    def encode_image(self, img: np.ndarray) -> np.ndarray:
        """SigLIP 风格：输出 256 维 patch token。"""
        if MOCK_MODE:
            return np.random.default_rng(int(img.sum())).standard_normal(256).astype(np.float32)
        raise NotImplementedError

    def encode_instruction(self, text: str) -> np.ndarray:
        if MOCK_MODE:
            # 简单 hash → 256 维
            h = abs(hash(text)) % (2**32)
            return np.random.default_rng(h).standard_normal(256).astype(np.float32)
        raise NotImplementedError

    def predict_action_chunk(self, image: np.ndarray, instruction: str, n_ode_steps: int = 10) -> np.ndarray:
        # 真实 Pi0：把 vision tokens + text tokens 拼接后输入 LLM，
        # LLM 输出 prefix KV，流匹配 head 消费 KV 迭代 denoise。
        # 这里 mock：直接用预设 target。
        x0 = np.random.default_rng(123).standard_normal((self.CHUNK_SIZE, self.ACTION_DIM)).astype(np.float32)
        chunk = np.stack([
            euler_solve(self.v_net, x0[t], n_steps=n_ode_steps)
            for t in range(self.CHUNK_SIZE)
        ])
        return chunk  # (50, 7)


# ---------- main ----------
def main() -> None:
    print("=== Pi0 — VLA with Flow Matching ===\n")
    print("Architecture:")
    print("  Vision encoder (SigLIP-400M)  -> 256 patch tokens")
    print("  Text encoder   (T5-XXL)       -> text tokens")
    print("  LLM backbone   (Pi0: 3B)      -> KV cache")
    print("  Flow head      (50Hz x 7DOF)   -> 50-step action chunk\n")

    rng = np.random.default_rng(0)
    fake_image = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    pi0 = Pi0Mini()
    chunk = pi0.predict_action_chunk(fake_image, "pick up the red cup")
    print(f"[Pi0] Predicted action chunk shape: {chunk.shape}  (chunk=50, dim=7)")
    print(f"[Pi0] First 3 actions:\n{np.round(chunk[:3], 3)}")

    # 训练损失 sanity check
    x0 = np.random.default_rng(0).standard_normal((4, 50, 7)).astype(np.float32)
    x1 = np.random.default_rng(1).standard_normal((4, 50, 7)).astype(np.float32)
    pred_v = (x1 - x0) * 0.5  # bad prediction
    print(f"\n[Pi0] Flow-matching loss (should drop toward 0 as v_net improves):")
    print(f"        current loss = {flow_matching_loss(x0, x1, pred_v):.4f}")
    print()
    print("OK")


if __name__ == "__main__":
    main()
