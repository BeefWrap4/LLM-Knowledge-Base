# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.3.3 HIL-SERL — 人在回路样本高效 RL
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: numpy
# run: MOCK_MODE=1 python 07_hil_serl.py
# expected_runtime: <2s
# expected_output: HIL-SERL 流程、reward model 学习、replay buffer 统计
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.3.3
# Interview hooks:
#   1. HIL-SERL 与 HIL-SERL (Luo et al. 2024) 的核心区别？人类干预信号如何被利用？
#   2. SERL 中 reward model 为什么用二分类（成功/失败）而不是回归？
#   3. 人类示范数据与 RL 探索数据如何混合？replay ratio 如何设置？

"""
HIL-SERL (Human-in-the-Loop Sample-Efficient RL) 简化演示。

核心思想：
  - 用人类示范做 behavior cloning 预训练
  - 部署到真实机器人后，人类可在失败时接管
  - 接管轨迹作为正样本 + 失败轨迹作为负样本，训练 reward classifier
  - 用该 reward 跑离线 RL (IQL / CQL) 微调策略
"""

import os
import numpy as np
from collections import deque


MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"


# ---------- 1. Replay Buffer ----------
class ReplayBuffer:
    """混合 buffer：(demonstrations ∪ agent_rollouts) + (success/failure labels)。"""

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.demos = deque(maxlen=capacity)
        self.rollouts_success = deque(maxlen=capacity)
        self.rollouts_failure = deque(maxlen=capacity)

    def add_demo(self, transition: dict):
        self.demos.append(transition)

    def add_rollout(self, transition: dict, success: bool):
        if success:
            self.rollouts_success.append(transition)
        else:
            self.rollouts_failure.append(transition)

    def stats(self) -> dict:
        return {
            "n_demos": len(self.demos),
            "n_success": len(self.rollouts_success),
            "n_failure": len(self.rollouts_failure),
            "total": len(self.demos) + len(self.rollouts_success) + len(self.rollouts_failure),
        }


# ---------- 2. 二分类 Reward Model ----------
class RewardClassifier:
    """P(success | obs, action) —— 来自人类接管 / 失败数据。"""

    def __init__(self, feat_dim: int = 32):
        rng = np.random.default_rng(0)
        self.W = rng.standard_normal((feat_dim, 1)).astype(np.float32) * 0.1
        self.b = np.zeros(1, dtype=np.float32)

    def predict_proba(self, feat: np.ndarray) -> float:
        logit = float(feat @ self.W + self.b)
        return 1.0 / (1.0 + np.exp(-logit))

    def update(self, feats: np.ndarray, labels: np.ndarray, lr: float = 0.01):
        # 简单梯度下降（BCE loss）
        for x, y in zip(feats, labels):
            p = self.predict_proba(x)
            grad = (p - y) * x
            self.W -= lr * grad[:, None]
            self.b -= lr * (p - y)


# ---------- 3. 人在回路 rollout ----------
def human_in_the_loop_rollout(
    policy,
    env,
    reward_clf: RewardClassifier,
    n_steps: int = 200,
    intervention_threshold: float = 0.3,
    seed: int = 0,
):
    """完整 HIL-SERL rollout：策略执行；人类在 reward 低时接管。"""
    rng = np.random.default_rng(seed)
    obs = env.reset()
    total_reward = 0.0
    human_interventions = 0
    buffer = ReplayBuffer(capacity=1000)
    feats_batch, labels_batch = [], []

    for t in range(n_steps):
        # 1) 策略选 action
        action = policy.act(obs)

        # 2) 环境 step
        next_obs, env_r, done, info = env.step(action)

        # 3) reward classifier 估计（实际可基于 obs+action 特征）
        feat = np.concatenate([obs[:16], action[:16]]).astype(np.float32)
        feat = np.pad(feat, (0, max(0, 32 - len(feat))))[:32]
        r_hat = reward_clf.predict_proba(feat)

        # 4) 人类干预判定：策略很可能失败
        if r_hat < intervention_threshold:
            human_action = info.get("expert_action", action)
            action = human_action
            human_interventions += 1
            label = 1  # 接管 = 正样本
        else:
            label = 0  # 自主完成

        buffer.add_rollout({"obs": obs, "action": action, "next_obs": next_obs,
                            "r_env": env_r, "r_hat": r_hat},
                           success=(label == 1))
        feats_batch.append(feat)
        labels_batch.append(label)
        total_reward += env_r

        obs = next_obs
        if done:
            break

    # 在线更新 reward model
    reward_clf.update(np.array(feats_batch), np.array(labels_batch, dtype=np.float32), lr=0.005)
    return total_reward, human_interventions, buffer.stats()


# ---------- 4. Mock 环境 / 策略 ----------
class MockEnv:
    def __init__(self, dim: int = 32, seed: int = 0):
        self.dim = dim
        self.state = None
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = self.rng.standard_normal(self.dim).astype(np.float32)
        return self.state

    def step(self, action: np.ndarray):
        # 简单 dynamics：把 action 填充/截断到 state dim
        a = np.zeros(self.dim, dtype=np.float32)
        a[: min(len(action), self.dim)] = action[: self.dim]
        self.state += 0.1 * a
        # 任务目标：前 8 维接近 1
        err = float(np.linalg.norm(self.state[:8] - 1.0))
        reward = max(0.0, 1.0 - 0.1 * err)
        done = err < 0.1
        info = {"expert_action": self.rng.standard_normal(len(action)).astype(np.float32) * 0.1}
        return self.state, reward, done, info


class MockPolicy:
    def __init__(self, dim: int = 32, action_dim: int = 7, seed: int = 0):
        self.W = np.random.default_rng(seed).standard_normal((dim, action_dim)).astype(np.float32) * 0.05
        self.dim = dim
        self.a_dim = action_dim

    def act(self, obs: np.ndarray) -> np.ndarray:
        a = obs @ self.W
        return np.tanh(a).astype(np.float32)


# ---------- main ----------
def main() -> None:
    print("=== HIL-SERL — Human-in-the-Loop Sample-Efficient RL ===\n")
    print("Pipeline:")
    print("  1) BC pre-train on demos (~50 trajectories)")
    print("  2) Deploy + rollouts; human intervenes on low reward")
    print("  3) Train reward classifier (success / failure)")
    print("  4) Offline RL (IQL/CQL) on combined buffer")
    print()

    env = MockEnv(seed=0)
    policy = MockPolicy(dim=32, action_dim=7, seed=0)
    reward_clf = RewardClassifier(feat_dim=32)

    print("[Phase 1] Rollout 1 (cold start, more interventions expected):")
    ret1, inter1, stats1 = human_in_the_loop_rollout(policy, env, reward_clf, n_steps=100, seed=0)
    print(f"  return={ret1:.2f}, human interventions={inter1}, buffer={stats1}")

    print("\n[Phase 2] Rollout 2 (reward classifier improved, fewer interventions):")
    ret2, inter2, stats2 = human_in_the_loop_rollout(policy, env, reward_clf, n_steps=100, seed=1)
    print(f"  return={ret2:.2f}, human interventions={inter2}, buffer={stats2}")

    print(f"\n[Summary] Intervention count dropped: {inter1} → {inter2}")
    print(f"          Total buffer transitions: {stats2['total']}")
    print()
    print("OK")


if __name__ == "__main__":
    main()
