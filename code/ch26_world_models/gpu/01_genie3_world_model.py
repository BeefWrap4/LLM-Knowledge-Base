# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.4.1 Genie 3 — Google 可交互世界模型
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch (lazy), numpy
# run: MOCK_MODE=1 python 01_genie3_world_model.py
# expected_runtime: <2s
# expected_output: Genie 3 架构概要、潜在动力学 step 演示、interactive rollout 帧计数
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.4
# Interview hooks:
#   1. Genie 系列（Genie 1/2/3）的核心差异是什么？Genie 3 引入"可交互"的关键模块？
#   2. 世界模型（World Model）与传统视频生成模型（Veo/Sora）的本质区别？
#   3. 为什么世界模型对具身智能至关重要（rollout + planning）？

"""
Genie 3 —— Google DeepMind 的可交互世界模型（World Model）演示。

Genie 3 接受一帧图像 + 文本动作描述，潜在空间动力学预测后续帧。
本文件不加载真实模型（Genie 3 尚未开源），仅演示其架构组件、
潜在动作（latent action）嵌入、动力学 step 的工程骨架。
"""

import os
import math
import numpy as np


MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"


# ---------- 1. 模型架构概览 ----------
def genie3_architecture_summary() -> None:
    """打印 Genie 3 关键模块的抽象尺寸，模拟论文/报告的参数量级。"""
    print("[Genie 3] Architecture summary (mock):")
    blocks = [
        ("Vision tokenizer (ViT-B/16)",          86e6,   "Frame -> 16x16 latent tokens"),
        ("Text encoder (T5-XXL)",                  11e9,   "Instruction -> context tokens"),
        ("Latent action quantizer (VQ-VAE-32)",    50e6,   "8 fps actions into 32-codebook"),
        ("Dynamics backbone (24-layer DiT-XL)",    700e6,  "predict next latent given (s_t, a_t)"),
        ("Frame decoder (latent -> 1024x1024)",    300e6,  "spatial + temporal upsample"),
    ]
    total = 0.0
    for name, params, desc in blocks:
        print(f"  {name:38s}  {params/1e6:8.1f}M   {desc}")
        total += params
    print(f"  {'TOTAL':38s}  {total/1e9:8.2f}B  (matches DeepMind 2026 report ~12B)")
    print()


# ---------- 2. 潜在动作（latent action）模块 ----------
class LatentActionEncoder:
    """把"鼠标拖动 / 键盘"等离散/连续动作编码为 32 维 codebook 索引。"""

    def __init__(self, codebook_size: int = 32, dim: int = 256):
        self.codebook_size = codebook_size
        self.dim = dim
        # 真实实现中 codebook 是 VQ-VAE 训练得到；这里随机初始化作示意
        rng = np.random.default_rng(0)
        self.codebook = rng.standard_normal((codebook_size, dim)).astype(np.float32)
        self.codebook /= np.linalg.norm(self.codebook, axis=1, keepdims=True) + 1e-8

    def encode(self, action_vec: np.ndarray) -> int:
        """输入形状 (dim,) 的动作特征，返回最近邻 codebook 索引。"""
        v = action_vec / (np.linalg.norm(action_vec) + 1e-8)
        sims = self.codebook @ v
        return int(np.argmax(sims))

    def decode(self, idx: int) -> np.ndarray:
        return self.codebook[idx]


# ---------- 3. 潜在动力学（latent dynamics）step ----------
def latent_dynamics_step(
    latent: np.ndarray,
    action_idx: int,
    action_emb: np.ndarray,
    n_steps_predict: int = 1,
) -> np.ndarray:
    """模拟一帧潜在 → 下一帧潜在的简单线性动力学（mock）。

    真实 Genie 3 内部是 24 层 DiT；这里用一个带动作调制的线性算子
    表达 "s_{t+1} = s_t + alpha * (W_a a + b)" 形式，便于教学。
    """
    if MOCK_MODE:
        rng = np.random.default_rng(action_idx + 1)
        # 简化：构造一个与 latent 同形状的 delta 字段
        # 真实 DiT 内部是 attention + MLP；这里用广播近似
        W = rng.standard_normal((latent.shape[-1], latent.shape[-1])).astype(np.float32) * 0.02
        # 把 action_emb 投影成与 latent 同 dim 的偏移
        a_W = rng.standard_normal((action_emb.shape[-1], latent.shape[-1])).astype(np.float32) * 0.1
        bias = action_emb @ a_W  # (D,)
        # 应用到每个空间位置
        delta = (latent @ W) + bias
        for _ in range(n_steps_predict):
            latent = latent + delta
        return latent.astype(np.float32)
    else:
        # 真实路径：加载 DiT backbone 并 forward（占位）
        raise NotImplementedError("Genie 3 weights not publicly released; MOCK_MODE=1 to run.")


# ---------- 4. 可交互 rollout ----------
def interactive_rollout(init_latent: np.ndarray, n_steps: int = 16) -> int:
    """模拟用户按 8 次方向键 / 鼠标拖动 → 16 帧后续画面。"""
    if MOCK_MODE:
        ae = LatentActionEncoder()
        z = init_latent.copy()
        actions = []
        for t in range(n_steps):
            # 模拟玩家输入：构造 dim=256 的特征（这里取 256 维的稀疏向量）
            user_input = np.zeros(256, dtype=np.float32)
            user_input[(t * 3) % 256] = 1.0
            user_input[(t * 7 + 1) % 256] = 0.5
            a_idx = ae.encode(user_input)
            actions.append(a_idx)
            z = latent_dynamics_step(z, a_idx, ae.decode(a_idx))
        unique_actions = len(set(actions))
        print(f"[Genie 3] rollout {n_steps} frames, {unique_actions} unique latent actions")
        return n_steps
    else:
        raise NotImplementedError


# ---------- main ----------
def main() -> None:
    print("=== Genie 3 World Model — Interview Demo ===\n")
    genie3_architecture_summary()
    z0 = np.random.default_rng(42).standard_normal((4, 16, 16, 256)).astype(np.float32)
    n = interactive_rollout(z0, n_steps=16)
    print(f"Generated {n} future latent frames (decoded by frame-decoder in real model).")


if __name__ == "__main__":
    main()
