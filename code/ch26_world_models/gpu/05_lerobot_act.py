# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.3.1 LeRobot ACT — Action Chunking Transformer
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch (lazy), numpy
# run: MOCK_MODE=1 python 05_lerobot_act.py
# expected_runtime: <2s
# expected_output: ACT 架构、CVAE 训练目标、action chunk 解码
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.3.1
# Interview hooks:
#   1. ACT (Action Chunking Transformer) 的核心思想？为什么分块预测比单步好？
#   2. CVAE 编码器在训练时引入，在推理时丢弃 —— 这样做的好处？
#   3. LeRobot 中 ACT 与 Diffusion Policy 的选择标准？精度 vs 多模态？

"""
LeRobot ACT (Action Chunking Transformer) 简化实现。

源自 ALOHA 论文 "Learning Fine-Grained Bimanual Manipulation" (Zhao et al., 2023)。
LeRobot (HuggingFace) 提供开箱即用训练脚本；本文件演示其核心组件：
  - CVAE encoder（仅训练时使用）
  - Transformer decoder（chunks of K future actions）
  - L1 regression head
"""

import os
import numpy as np


MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"


# ---------- 1. CVAE Encoder (训练时) ----------
class CVAEEncoder:
    """把 (obs, action_chunk) → latent style z (dim=32)。"""

    LATENT_DIM = 32

    def __init__(self, state_dim: int = 14, action_dim: int = 14, chunk: int = 100):
        # state = 2 x 7 (ALOHA 双臂关节角)
        # action chunk = chunk x 14
        rng = np.random.default_rng(1)
        d_in = state_dim + action_dim * chunk
        self.W = rng.standard_normal((d_in, 2 * self.LATENT_DIM)).astype(np.float32) * 0.001
        self.d_in = d_in

    def encode(self, state: np.ndarray, action_chunk: np.ndarray):
        x = np.concatenate([state, action_chunk.flatten()])
        out = x @ self.W
        mu, logvar = out[: self.LATENT_DIM], out[self.LATENT_DIM:]
        return mu, logvar

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * logvar)
        eps = np.random.default_rng().standard_normal(mu.shape).astype(np.float32)
        return mu + eps * std


# ---------- 2. Transformer Decoder (CVAE) ----------
class ACTDecoder:
    """简化 Transformer decoder：输入 (state, z) → 输出 (chunk, action_dim)。"""

    def __init__(self, state_dim: int = 14, latent_dim: int = 32, action_dim: int = 14, chunk: int = 100):
        rng = np.random.default_rng(2)
        # 一层 attention + FFN（教学 mock）
        self.W_q = rng.standard_normal((state_dim + latent_dim, 64)).astype(np.float32) * 0.1
        self.W_k = rng.standard_normal((state_dim + latent_dim, 64)).astype(np.float32) * 0.1
        self.W_v = rng.standard_normal((state_dim + latent_dim, 64)).astype(np.float32) * 0.1
        self.W_o = rng.standard_normal((64, action_dim)).astype(np.float32) * 0.1
        self.chunk = chunk

    def forward(self, state: np.ndarray, z: np.ndarray) -> np.ndarray:
        # 把 (state, z) 复制 chunk 份当 query，输出 chunk 个 action
        cond = np.concatenate([state, z])  # (D,)
        seq = np.tile(cond, (self.chunk, 1))  # (chunk, D)
        Q = seq @ self.W_q
        K = seq @ self.W_k
        V = seq @ self.W_v
        attn = np.exp((Q @ K.T) / np.sqrt(64))
        attn /= attn.sum(axis=-1, keepdims=True) + 1e-8
        out = (attn @ V) @ self.W_o
        return out.astype(np.float32)


# ---------- 3. ACT 模型 ----------
class ACTPolicy:
    """Action Chunking Transformer for ALOHA-style bimanual manipulation."""

    CHUNK = 100       # 100 步 ≈ 5s @ 20Hz
    STATE_DIM = 14    # 2 x 7 (bimanual joints)
    ACTION_DIM = 14

    def __init__(self):
        self.encoder = CVAEEncoder(self.STATE_DIM, self.ACTION_DIM, self.CHUNK)
        self.decoder = ACTDecoder(self.STATE_DIM, CVAEEncoder.LATENT_DIM, self.ACTION_DIM, self.CHUNK)

    def train_step_loss(self, state: np.ndarray, action_chunk: np.ndarray) -> dict:
        mu, logvar = self.encoder.encode(state, action_chunk)
        z = self.encoder.reparameterize(mu, logvar)
        pred = self.decoder.forward(state, z)
        recon = float(np.mean(np.abs(pred - action_chunk)))   # L1
        kl = float(-0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar)))
        return {"recon_l1": round(recon, 4), "kl": round(kl, 4), "total": round(recon + 1e-3 * kl, 4)}

    def select_action(self, state: np.ndarray) -> np.ndarray:
        # 推理时：z 置 0
        z = np.zeros(CVAEEncoder.LATENT_DIM, dtype=np.float32)
        chunk = self.decoder.forward(state, z)
        return chunk[0]  # 只取第一步执行（temporal ensemble 也可）


# ---------- main ----------
def main() -> None:
    print("=== LeRobot ACT — Action Chunking Transformer ===\n")
    print("Architecture:")
    print("  State (14) + CVAE-z (32)  -> Transformer Decoder")
    print("  -> chunk=100 future actions (L1 regression)")
    print("  -> 执行前 K 步 + temporal ensemble\n")

    rng = np.random.default_rng(42)
    state = rng.standard_normal(14).astype(np.float32) * 0.1
    target_chunk = rng.standard_normal((100, 14)).astype(np.float32) * 0.05

    policy = ACTPolicy()
    losses = policy.train_step_loss(state, target_chunk)
    print(f"[ACT] Train losses: {losses}")
    action = policy.select_action(state)
    print(f"[ACT] First action:  dim={action.shape[0]}, range=[{action.min():.3f}, {action.max():.3f}]")
    print()
    print("OK")


if __name__ == "__main__":
    main()
