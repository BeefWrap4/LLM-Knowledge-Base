# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.2.2 GR00T N1.5 — NVIDIA 通用机器人基础模型
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch (lazy), numpy
# run: MOCK_MODE=1 python 04_groot_n15_vla.py
# expected_runtime: <2s
# expected_output: GR00T N1.5 架构、Eagle-2.5 VLM 接口、跨本体动作适配器
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.2.2
# Interview hooks:
#   1. GR00T N1.5 如何处理"跨本体"（不同机器人形态）的动作空间异构？
#   2. Eagle-2.5 VLM 与 SigLIP 相比，多模态融合在 VLA 任务上有什么优势？
#   3. 训练 GR00T 这类模型需要多大规模的数据？仿真 : 真实比例？

"""
NVIDIA GR00T N1.5 —— 跨本体通用机器人基础模型（Generalist Robot Model）。

特点：
  - 基座：Eagle-2.5 VLM (5B)
  - 跨本体：不同机器人（Franka、Humanoid、ALOHA）共享 latent action space
  - 训练数据：仿真 (Isaac Sim) + 真实遥操作
  - 推理：消费级 H100 / RTX 4090 即可 fine-tune
"""

import os
import numpy as np


MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"


# ---------- 1. 跨本体动作空间统一 ----------
class EmbodimentActionAdapter:
    """把不同机器人的异构动作空间投影到统一 latent。

    Franka Panda: 7-DOF arm + 1 gripper       -> 8 dim
    Humanoid (Unitree H1):  19 joints          -> 19 dim
    ALOHA bimanual:        2 x 6-DOF + 2 grip -> 14 dim

    统一 latent dim = 32。
    """

    NATIVE_DIMS = {
        "franka": 8,
        "humanoid_h1": 19,
        "aloha_bimanual": 14,
    }
    LATENT_DIM = 32

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.projs = {}
        for name, dim in self.NATIVE_DIMS.items():
            # 实际训练得到；这里随机初始化
            self.projs[name] = rng.standard_normal((dim, self.LATENT_DIM)).astype(np.float32) * 0.1

    def encode(self, name: str, action: np.ndarray) -> np.ndarray:
        if name not in self.projs:
            raise ValueError(f"Unknown embodiment: {name}")
        return action @ self.projs[name]

    def decode(self, name: str, latent: np.ndarray) -> np.ndarray:
        # pinv 解码回原空间
        W = self.projs[name]
        return latent @ np.linalg.pinv(W)


# ---------- 2. Eagle-2.5 VLM 接口（mock） ----------
class EagleVLMBackbone:
    """Eagle-2.5 视觉-语言 backbone —— 把 (image, text) → hidden states。"""

    HIDDEN_DIM = 2048
    NUM_LAYERS = 32

    def __init__(self):
        print(f"  [Eagle-2.5] Loaded mock backbone: hidden={self.HIDDEN_DIM}, layers={self.NUM_LAYERS}")

    def forward(self, image: np.ndarray, text: str) -> np.ndarray:
        """真实情况：把 1024 个 vision token + 128 个 text token 输入 ViT-LLM。
        Mock：返回固定长度的 hidden states。"""
        n_img_tokens = 256
        n_txt_tokens = 32
        return np.random.default_rng(hash(text) & 0xFFFF).standard_normal(
            (n_img_tokens + n_txt_tokens, self.HIDDEN_DIM)
        ).astype(np.float32)


# ---------- 3. GR00T mini ----------
class GR00TN15:
    """GR00T N1.5 简化版：Eagle-2.5 + cross-embodiment action head。"""

    def __init__(self):
        self.vlm = EagleVLMBackbone()
        self.adapter = EmbodimentActionAdapter()
        # VLM hidden -> latent action 的投影头（mock）
        rng = np.random.default_rng(2026)
        self.head = rng.standard_normal(
            (EagleVLMBackbone.HIDDEN_DIM, EmbodimentActionAdapter.LATENT_DIM)
        ).astype(np.float32) * 0.01

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        embodiment: str = "franka",
    ) -> np.ndarray:
        h = self.vlm.forward(image, instruction)
        # 用 mean pool 简化
        pooled = h.mean(axis=0)
        latent = pooled @ self.head
        # 解码到目标本体动作
        action = self.adapter.decode(embodiment, latent)
        return action


# ---------- main ----------
def main() -> None:
    print("=== NVIDIA GR00T N1.5 — Cross-Embodiment VLA ===\n")

    groot = GR00TN15()
    fake_img = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    instr = "place the screwdriver in the box"

    print("Cross-embodiment action inference:")
    for emb in EmbodimentActionAdapter.NATIVE_DIMS:
        a = groot.predict(fake_img, instr, embodiment=emb)
        print(f"  {emb:18s}  predicted action dim={a.shape[0]:2d}  (e.g. {np.round(a[:3], 2)})")

    # 编码 / 解码一致性测试
    adapter = EmbodimentActionAdapter()
    franka_act = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.4, 0.5, 0.8], dtype=np.float32)
    z = adapter.encode("franka", franka_act)
    rec = adapter.decode("franka", z)
    err = float(np.linalg.norm(rec - franka_act))
    print(f"\n[Adapter] encode→decode reconstruction error: {err:.4f} (should be ~0)")
    print()


if __name__ == "__main__":
    main()
