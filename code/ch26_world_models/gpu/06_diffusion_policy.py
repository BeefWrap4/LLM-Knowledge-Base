# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.3.2 Diffusion Policy — Chi 等 2023
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch (lazy), numpy
# run: MOCK_MODE=1 python 06_diffusion_policy.py
# expected_runtime: <2s
# expected_output: DDPM 训练步 + DDIM 推理轨迹 + 动作可视化
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.3.2
# Interview hooks:
#   1. Diffusion Policy 为什么能处理多模态动作分布（vs Gaussian/L2 单峰）？
#   2. 与 ACT 的对比：DDPM 去噪过程如何解释为"迭代精化"？
#   3. 推理时使用 DDIM 多少步合适？步数与精度的权衡？

"""
Diffusion Policy (Chi et al., RSS 2023) 简化实现。

核心思想：
  - 训练：把 (obs, action_chunk) 对看作 (condition, target)，用 DDPM 学习
    条件去噪网络 ε_θ(a_t, t, obs)
  - 推理：从 a_T ~ N(0, I) 开始迭代去噪 K 步（DDIM）
  - 支持多模态动作分布（重要优势）
"""

import os
import numpy as np


MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"


# ---------- 1. 噪声调度 ----------
def linear_beta_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)


def make_ddpm_constants(T: int = 100):
    betas = linear_beta_schedule(T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas).astype(np.float32)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars, "T": T}


# ---------- 2. 简化 ε-network ----------
class MockEpsilonNet:
    """ε_θ(a_t, t, obs) —— 实际是 1D U-Net / Transformer。"""

    def __init__(self, action_dim: int = 14, cond_dim: int = 14, hidden: int = 128):
        rng = np.random.default_rng(0)
        d_in = action_dim + cond_dim + 1  # +1 for time
        self.W1 = rng.standard_normal((d_in, hidden)).astype(np.float32) * 0.1
        self.W2 = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.1
        self.W3 = rng.standard_normal((hidden, action_dim)).astype(np.float32) * 0.1
        self.action_dim = action_dim
        self.cond_dim = cond_dim

    def __call__(self, a_t: np.ndarray, t: float, cond: np.ndarray) -> np.ndarray:
        # a_t: (action_dim,)  cond: (cond_dim,)
        a_t = np.atleast_1d(a_t).astype(np.float32)
        cond = np.atleast_1d(cond).astype(np.float32)
        # 用 a_t 的均值代表当前 action（教学简化；真实实现是逐 token 预测）
        a_feat = np.array([float(a_t.mean())] * self.action_dim, dtype=np.float32)
        x = np.concatenate([a_feat, cond, [t / 1000.0]]).astype(np.float32)
        h = np.tanh(x @ self.W1)
        h = np.tanh(h @ self.W2)
        return h @ self.W3


# ---------- 3. 训练一步 ----------
def ddpm_train_step(eps_net: MockEpsilonNet, a0: np.ndarray, cond: np.ndarray, consts: dict) -> float:
    """简化 DDPM 训练：随机采样 t，计算 L = ||ε - ε_θ(a_t, t, cond)||^2。"""
    T = consts["T"]
    ab = consts["alpha_bars"]
    B = a0.shape[0]
    t = np.random.randint(0, T, size=B).astype(np.int64)
    eps = np.random.default_rng().standard_normal(a0.shape).astype(np.float32)
    ab_t = ab[t][:, None, None]  # (B, 1, 1) for (B, T, D) shape
    a_t = np.sqrt(ab_t) * a0 + np.sqrt(1 - ab_t) * eps
    # 真实实现：批量 forward ε_θ；这里取 chunk 整体均值做 loss（教学简化）
    losses = []
    a_dim = eps_net.action_dim
    for i in range(B):
        flat_a = a_t[i].reshape(-1)
        flat_eps = eps[i].reshape(-1)
        pred = eps_net(np.array([float(flat_a.mean())]), float(t[i]), cond[i])
        target = np.full(a_dim, float(flat_eps.mean()), dtype=np.float32)
        losses.append(float(np.mean((pred - target) ** 2)))
    return float(np.mean(losses))


# ---------- 4. DDIM 推理 ----------
def ddim_sample(eps_net: MockEpsilonNet, cond: np.ndarray, action_dim: int = 14,
                chunk: int = 16, n_steps: int = 10, consts: dict = None) -> np.ndarray:
    """从 a_T ~ N(0, I) 出发 DDIM 采样 n_steps 步。"""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((chunk, action_dim)).astype(np.float32)
    ab = consts["alpha_bars"]
    # DDIM 时间步（均匀子集）
    timesteps = np.linspace(0, consts["T"] - 1, n_steps, dtype=int)[::-1]
    for i, t in enumerate(timesteps):
        ab_t = ab[t]
        ab_prev = ab[timesteps[i + 1]] if i + 1 < len(timesteps) else 1.0
        # 逐 chunk 步去噪（实际是 batched）
        for k in range(chunk):
            eps = eps_net(a[k], float(t), cond)
            # 预测 a0
            a0_hat = (a[k] - np.sqrt(1 - ab_t) * eps) / (np.sqrt(ab_t) + 1e-8)
            # DDIM 更新
            a[k] = np.sqrt(ab_prev) * a0_hat + np.sqrt(1 - ab_prev) * eps
    return a


# ---------- main ----------
def main() -> None:
    print("=== Diffusion Policy (DDPM/DDIM) ===\n")
    print("Training: ε_θ predicts noise on action_chunk, conditioned on obs.")
    print("Inference: 10-20 DDIM steps from N(0,I) → clean action chunk.\n")

    consts = make_ddpm_constants(T=100)
    print(f"[DDPM] beta schedule: {consts['betas'][:3].round(4).tolist()} ... "
          f"alpha_bar end: {consts['alpha_bars'][-1]:.4f}")

    eps_net = MockEpsilonNet(action_dim=14, cond_dim=14)
    cond = np.random.default_rng(0).standard_normal((4, 14)).astype(np.float32) * 0.1
    a0   = np.random.default_rng(1).standard_normal((4, 16, 14)).astype(np.float32) * 0.3

    print("\n[Train] DDPM step losses (should hover around 0.5-1.5 for random init):")
    for step in range(3):
        loss = ddpm_train_step(eps_net, a0, cond, consts)
        print(f"  step {step+1}: L_simple = {loss:.4f}")

    print("\n[Inference] DDIM 10-step sampling...")
    cond_single = cond[0]
    sample = ddim_sample(eps_net, cond_single, action_dim=14, chunk=16,
                         n_steps=10, consts=consts)
    print(f"  generated action chunk: shape={sample.shape}, "
          f"range=[{sample.min():.3f}, {sample.max():.3f}]")
    print()
    print("OK")


if __name__ == "__main__":
    main()
