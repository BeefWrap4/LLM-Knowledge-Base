# ---
# chapter: 27
# topic: MCTS + PRM guided search (AlphaProof-style)
# section: 27.5.1 MCTS + PRM
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: numpy
# run: python 11_mcts_prm.py
# expected_runtime: <2s
# expected_output: 打印 MCTS 搜索过程 + 最优路径
# ---
# See: ../tutorial/27_推理模型与Test-Time_Compute.md §27.5.1
# Interview hooks:
#   1. UCB 公式各项含义？exploration constant c 如何调？
#   2. MCTS + PRM 与 Best-of-N 比，优势在哪？代价？
#   3. AlphaProof / rStar-Math 用了哪些 PRM 训练数据？
"""MCTS (Monte Carlo Tree Search) with PRM 引导。

四步循环: Selection → Expansion → Simulation → Backprop
PRM 作为 Expansion 时的先验 + Simulation 时的 rollout 价值。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCTSNode:
    state: str
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 1.0           # PRM 给的先验
    is_terminal: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / max(1, self.visits)

    def ucb(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return (self.value_sum / self.visits) + c * self.prior * math.sqrt(
            parent_visits
        ) / (1 + self.visits)


def mock_prm(state: str) -> float:
    """模拟 PRM：state 越接近答案 prior 越高。"""
    if "answer=correct" in state:
        return 1.0
    if "answer=wrong" in state:
        return 0.05
    # 启发式：状态越深 prior 略降
    depth = state.count(">")
    return max(0.1, 0.9 - 0.1 * depth) * random.uniform(0.8, 1.2)


def mock_expand(node: MCTSNode, branching: int = 3) -> None:
    """生成子节点，PRM 给先验。"""
    if node.is_terminal or node.children:
        return
    for i in range(branching):
        s = f"{node.state} > step{i}"
        # 末位 step2 模拟"正确"答案
        if "step2" in s and random.random() < 0.4:
            s = s.replace("step2", "answer=correct")
            child = MCTSNode(s, parent=node, prior=mock_prm(s),
                             is_terminal=True)
        else:
            child = MCTSNode(s, parent=node, prior=mock_prm(s))
        node.children.append(child)


def mcts_search(root_state: str, n_simulations: int = 50,
                c: float = 1.4) -> MCTSNode:
    root = MCTSNode(root_state, prior=1.0)

    for sim in range(n_simulations):
        node = root

        # Selection: 沿 UCB 走到叶
        while node.children:
            node = max(node.children, key=lambda n: n.ucb(c))

        # Expansion
        if not node.is_terminal:
            mock_expand(node, branching=3)
            if node.children:
                # 选 prior 最高的扩展
                node = max(node.children, key=lambda n: n.prior)

        # Simulation: 简化 — 直接用 PRM 当 rollout value
        reward = mock_prm(node.state)

        # Backprop
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += reward
            cur = cur.parent

    return root


def best_path(root: MCTSNode) -> MCTSNode:
    """从根走 visits 最大的子节点。"""
    cur = root
    while cur.children:
        cur = max(cur.children, key=lambda n: n.visits)
    return cur


def main() -> None:
    random.seed(42)
    root = mcts_search("Q", n_simulations=80)

    # 打印访问统计
    def visit_count(n: MCTSNode) -> int:
        return n.visits

    print("MCTS search tree (visits):")
    print(f"  root [{root.visits}]")

    def show(n: MCTSNode, depth: int = 1) -> None:
        for c in sorted(n.children, key=visit_count, reverse=True):
            tag = " [TERMINAL]" if c.is_terminal else ""
            print(f"  {'  ' * depth}- {c.state.split(' > ')[-1]:>10}"
                  f"  V={c.visits:>3}  v={c.value:.2f}"
                  f"  prior={c.prior:.2f}{tag}")
            if depth < 2:
                show(c, depth + 1)

    show(root)

    best = best_path(root)
    print(f"\nBest path: {best.state}")
    print(f"  visits={best.visits}  value={best.value:.3f}")

    print("\n--- 与 Best-of-N 对比 ---")
    print(f"  MCTS+PRM  : {root.visits} 次模拟 → 通常 N=64 BoN 更准但代价高")
    print(f"  Best-of-N : 独立采样 N 次，verifier 选最优")
    print(f"  混合策略  : MCTS 指导 prefix → N 个完成 → verifier 选最优")
    print("OK")


if __name__ == "__main__":
    main()
