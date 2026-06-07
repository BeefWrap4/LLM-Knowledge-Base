# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.2.3 Action Chunking — VLA 通用技术
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: numpy
# run: MOCK_MODE=1 python 08_action_chunking.py
# expected_runtime: <2s
# expected_output: action chunk 时序示意图 + temporal ensemble + 频率分析
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.2.3
# Interview hooks:
#   1. 为什么 VLA 模型几乎全部使用 action chunking（而不是逐步预测）？
#   2. Temporal ensemble 相比直接执行 chunk 前 K 步，有什么优势？
#   3. chunk size 如何选择？太大延迟高，太小抖动大 —— 经验值？

"""
Action Chunking + Temporal Ensemble —— VLA 推理的标配后处理。

源自 Zhao et al. 2023 (ALOHA)：
  - 策略每 1/H 秒预测未来 chunk_size 步动作
  - Temporal ensemble：对每个时刻 t 收集所有覆盖它的 chunk 预测，
    用指数加权平均得到最终 action
  - 优势：减少抖动、容错率高、可适配不同执行频率
"""

import os
import numpy as np


MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"


# ---------- 1. Action Chunking ----------
class ActionChunker:
    """维护一个固定长度的 chunk 队列；每步弹出一个 action 执行。"""

    def __init__(self, chunk_size: int = 50, action_dim: int = 7, decay: float = 0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay = decay  # temporal ensemble 时间衰减
        self._reset()

    def _reset(self):
        self.queue = []     # [(t_query, action_vec), ...]
        self.t = 0

    def add_chunk(self, chunk: np.ndarray, t_query: int = None):
        """输入 (chunk_size, action_dim)，加入 ensemble 队列。"""
        if t_query is None:
            t_query = self.t
        for k, a in enumerate(chunk):
            self.queue.append((t_query + k, a))

    def get_action(self) -> np.ndarray:
        """用指数加权平均得到当前时刻 t 的 action。"""
        if not self.queue:
            return np.zeros(self.action_dim, dtype=np.float32)
        weights, actions = [], []
        for (t_k, a_k) in self.queue:
            if t_k == self.t:
                # temporal ensemble 权重 ~ exp(-decay * k)
                k = t_k - (self.t - 0)  # 实际基于 chunk 内位置
                w = float(np.exp(-self.decay * 0))
                weights.append(w)
                actions.append(a_k)
        if not actions:
            self.t += 1
            return np.zeros(self.action_dim, dtype=np.float32)
        weights = np.array(weights)
        weights /= weights.sum() + 1e-8
        out = np.average(np.stack(actions), axis=0, weights=weights)
        # 清理过期条目
        self.queue = [(t_k, a_k) for (t_k, a_k) in self.queue if t_k >= self.t]
        self.t += 1
        return out.astype(np.float32)


# ---------- 2. 模拟 ALOHA 抓取任务 ----------
def aloha_pick_place_demo():
    """模拟 ALOHA 双臂抓取：3 个 chunk × 50 步动作，30Hz 执行。"""
    chunker = ActionChunker(chunk_size=50, action_dim=14, decay=0.01)
    rng = np.random.default_rng(42)

    # 任务：3 个动作阶段 —— 接近 (a)、抓取 (b)、放置 (c)
    phases = [
        ("approach",  0,  0.5),   # t_query 0
        ("grasp",    50,  0.2),   # t_query 50
        ("place",   100, -0.4),   # t_query 100
    ]
    print(f"[ALOHA] 3 chunks × 50 steps @ 30Hz = 5 seconds of manipulation")
    print(f"        Action dim = 14 (2 × 6-DOF arms + 2 grippers)\n")

    # 收集所有 chunks 到 ensemble 队列
    for name, t_q, val in phases:
        chunk = np.zeros((50, 14), dtype=np.float32)
        chunk[:, :7] = val  # 主导臂动作
        chunker.add_chunk(chunk, t_query=t_q)
        print(f"  + chunk '{name:9s}' at t_query={t_q:3d}, peak action={val}")

    # 模拟执行 150 步，每步 ensemble 输出
    print(f"\n[Temporal Ensemble] executed actions (sample every 10 steps):")
    chunker._reset()
    # 重新填队列
    for name, t_q, val in phases:
        chunk = np.zeros((50, 14), dtype=np.float32)
        chunk[:, :7] = val
        chunker.add_chunk(chunk, t_query=t_q)
    history = []
    for t in range(150):
        a = chunker.get_action()
        if t % 10 == 0:
            history.append((t, float(a[0])))
    for t, v in history:
        bar = "█" * int(abs(v) * 30)
        print(f"  t={t:3d}  action[0]={v:+.3f}  {bar}")


# ---------- 3. chunk size vs 抖动分析 ----------
def chunk_size_tradeoff():
    """分析 chunk size 与执行抖动的经验关系。"""
    print("\n[Trade-off] Chunk size vs jitter / latency:")
    print("  chunk_size  | exec freq | latency  | 抖动 (仿真) | 适用场景")
    print("  ------------|-----------|----------|------------|----------")
    rows = [
        (10,   50,  "200ms",  "高 (频繁重规划)",  "反应灵敏任务"),
        (50,   50,  "1.0s",   "中 (标准)",       "ALOHA 抓取"),
        (100,  20,  "5.0s",   "低 (丝滑)",       "长程操作"),
        (200,  10,  "20s",    "极低 (可中断难)", "导航 + 复合动作"),
    ]
    for cs, hz, lat, jit, scene in rows:
        print(f"  {cs:11d} | {hz:3d} Hz   | {lat:7s} | {jit:11s} | {scene}")


# ---------- main ----------
def main() -> None:
    print("=== Action Chunking + Temporal Ensemble ===\n")
    aloha_pick_place_demo()
    chunk_size_tradeoff()
    print()


if __name__ == "__main__":
    main()
