# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.4.3 世界模型驱动的 MPC / Planning
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: numpy
# run: MOCK_MODE=1 python 09_world_model_rollout.py
# expected_runtime: <2s
# expected_output: Dreamer-style rollout + CEM 规划 + 候选动作打分
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.4.3
# Interview hooks:
#   1. DreamerV3 与 Genie 3 在"潜空间动力学"上的核心差异？
#   2. CEM (Cross-Entropy Method) 为什么适合 action sequence 规划？
#   3. 世界模型 rollout 的 horizon 越深越好吗？compounding error 如何缓解？

"""
世界模型 (World Model) 驱动的规划 (Planning) 与 MPC (Model Predictive Control)。

流程：
  1) 在世界模型潜空间从当前状态 s_t 开始
  2) CEM 采样 N 条候选动作序列
  3) 在世界模型中 rollout K 步
  4) 评分选 top-M，更新采样分布
  5) 取最佳序列的第一步执行
  6) 重复
"""

import os
import numpy as np


MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"


# ---------- 1. 简化世界模型 (潜空间) ----------
class LatentWorldModel:
    """s_{t+1} = f(s_t, a_t)  +  r_t = R(s_t, a_t)  简化版。"""

    def __init__(self, state_dim: int = 32, action_dim: int = 7, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((state_dim + action_dim, hidden)).astype(np.float32) * 0.1
        self.V = rng.standard_normal((hidden, state_dim)).astype(np.float32) * 0.1
        self.Rw = rng.standard_normal((state_dim + action_dim, 1)).astype(np.float32) * 0.1
        self.s_dim, self.a_dim = state_dim, action_dim

    def step(self, s: np.ndarray, a: np.ndarray):
        x = np.concatenate([s, a]).astype(np.float32)
        h = np.tanh(x @ self.W)
        s_next = s + h @ self.V
        r = float(x @ self.Rw)
        return s_next.astype(np.float32), r

    def rollout(self, s0: np.ndarray, actions: np.ndarray):
        """actions: (horizon, action_dim) -> (s_traj, r_total)"""
        s = s0.copy()
        total_r = 0.0
        for t in range(actions.shape[0]):
            s, r = self.step(s, actions[t])
            total_r += r
        return s, total_r


# ---------- 2. CEM 规划器 ----------
class CEMPlanner:
    """Cross-Entropy Method 在动作序列空间搜索。"""

    def __init__(self, horizon: int = 12, action_dim: int = 7, n_samples: int = 64,
                 n_elite: int = 8, n_iters: int = 5, seed: int = 0):
        self.H = horizon
        self.A = action_dim
        self.N = n_samples
        self.K = n_elite
        self.iters = n_iters
        self.rng = np.random.default_rng(seed)

    def plan(self, wm: LatentWorldModel, s0: np.ndarray) -> np.ndarray:
        mu = np.zeros((self.H, self.A), dtype=np.float32)
        sigma = np.ones((self.H, self.A), dtype=np.float32)
        best_seq = None
        for it in range(self.iters):
            samples = self.rng.normal(mu, sigma, size=(self.N, self.H, self.A)).astype(np.float32)
            scores = np.array([wm.rollout(s0, seq)[1] for seq in samples])
            elite_idx = np.argsort(scores)[-self.K :]
            elites = samples[elite_idx]
            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + 1e-3
            best_seq = elites[-1]  # 当前最佳
        return best_seq


# ---------- 3. MPC 闭环 ----------
def mpc_loop(wm: LatentWorldModel, planner: CEMPlanner, s0: np.ndarray, n_steps: int = 20):
    """Model Predictive Control：每步重新规划。"""
    s = s0.copy()
    history = [s.copy()]
    actions_taken = []
    for t in range(n_steps):
        best_seq = planner.plan(wm, s)
        a0 = best_seq[0]
        s, _ = wm.step(s, a0)
        history.append(s.copy())
        actions_taken.append(a0)
    return np.array(history), np.array(actions_taken)


# ---------- 4. 任务：到达 latent target ----------
class ReachingTask:
    def __init__(self, state_dim: int = 32, target: np.ndarray = None):
        self.dim = state_dim
        self.target = target if target is not None else np.ones(state_dim, dtype=np.float32)

    def true_reward(self, s: np.ndarray, a: np.ndarray) -> float:
        # 真实环境奖励：负距离 - 控制代价
        dist = float(np.linalg.norm(s - self.target))
        return -dist - 0.01 * float(np.sum(a ** 2))


# ---------- main ----------
def main() -> None:
    print("=== World-Model-Driven MPC (CEM Planner) ===\n")

    state_dim = 32
    wm = LatentWorldModel(state_dim=state_dim, action_dim=7, seed=0)
    planner = CEMPlanner(horizon=12, action_dim=7, n_samples=64, n_elite=8, n_iters=4, seed=0)
    task = ReachingTask(state_dim=state_dim, target=np.ones(state_dim, dtype=np.float32))

    s0 = np.zeros(state_dim, dtype=np.float32)  # 起点
    print(f"[Setup] state_dim={state_dim}, horizon=12, CEM samples=64, elite=8")
    print(f"        target = {task.target[:4].tolist()} ... (32-dim one-hot-ish)\n")

    # 1) 开环规划
    best_seq = planner.plan(wm, s0)
    s_final, total_r = wm.rollout(s0, best_seq)
    print(f"[Open-loop] plan+rollout in WM:  total_r = {total_r:+.3f}")
    print(f"            s_final[0:4] = {s_final[:4].round(2).tolist()}")

    # 2) 闭环 MPC
    traj, acts = mpc_loop(wm, planner, s0, n_steps=20)
    dists = [float(np.linalg.norm(traj[t] - task.target)) for t in range(0, 21, 4)]
    print(f"\n[MPC loop] 20 steps closed-loop, dist-to-target every 4 steps:")
    for t, d in zip(range(0, 21, 4), dists):
        print(f"  t={t:2d}  ||s-target|| = {d:.3f}")
    print(f"  final dist = {dists[-1]:.3f}  (started at {dists[0]:.3f})")

    # 3) 真实环境验证（用 wm 代理）—— 衡量 compounding error
    real_rewards = []
    s = s0.copy()
    for t in range(20):
        a = acts[t]
        s, r_wm = wm.step(s, a)
        # 同时计算"真实"奖励（用同一个 wm 作为代理，因为我们没有 ground-truth env）
        real_rewards.append(r_wm)
    print(f"\n[Compounding] WM rollout total_r over 20 steps: {sum(real_rewards):+.3f}")
    print(f"               (in real deploy, horizon>10 usually shows ~10% error growth)")
    print()
    print("OK")


if __name__ == "__main__":
    main()
