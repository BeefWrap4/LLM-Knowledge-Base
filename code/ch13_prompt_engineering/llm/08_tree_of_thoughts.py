# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.2.5 Tree of Thoughts
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无外部依赖 (LLM 调用使用 mock)
# run: python 08_tree_of_thoughts.py
# expected_runtime: <1s
# expected_output: 打印 ToT 搜索过程与得分最高的状态
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.2.5
# Interview hooks:
# - CoT 与 ToT 的本质区别？为什么 ToT 适合 24 点等组合问题？
# - 评估函数 evaluate() 如何避免误导搜索方向？
# - BFS 与 DFS 在 ToT 中的取舍？

import random
from heapq import heappop, heappush


def call_llm(prompt: str) -> str:
    """模拟 LLM：根据请求类型返回不同内容。"""
    if prompt.startswith("评估"):
        # 评估分数：根据 prompt 长度做伪随机
        seed = sum(ord(c) for c in prompt) % 100
        return str(random.Random(seed).uniform(3, 9))
    # 生成思路：返回若干候选
    return "\n".join([f"思路 {i + 1}：用算式组合达成目标" for i in range(3)])


class TreeOfThoughts:
    """Tree of Thoughts 简版实现"""

    def __init__(self, branch_factor: int = 3, max_depth: int = 5):
        self.branch_factor = branch_factor  # 每节点分支数
        self.max_depth = max_depth  # 最大搜索深度

    def generate_thoughts(self, state: str, k: int) -> list[str]:
        """从当前状态生成 k 个候选思考步骤"""
        prompt = f"基于当前进展：{state}\n请提出 {k} 个不同的下一步思路（每行一个）："
        response = call_llm(prompt)
        return [t.strip() for t in response.split("\n") if t.strip()][:k]

    def evaluate(self, state: str) -> float:
        """评估当前思考路径的 promising 程度（0-1）"""
        prompt = f"评估以下解题进展的可行性（只输出 0-10 的数字）：\n{state}"
        score = float(call_llm(prompt).strip()) / 10
        return score

    def search(self, initial_state: str) -> str:
        """BFS + 评估函数进行树搜索"""
        # 优先队列：(负分, 深度, 计数器, 状态) — 计数器用于 tie-break 避免比较 str
        queue = [(-self.evaluate(initial_state), 0, 0, initial_state)]
        best_state = initial_state
        best_score = -1
        counter = 1

        while queue:
            neg_score, depth, _, state = heappop(queue)
            score = -neg_score

            if score > best_score:
                best_score = score
                best_state = state

            if depth >= self.max_depth:
                continue

            # 生成子节点
            for thought in self.generate_thoughts(state, self.branch_factor):
                new_state = state + "\n" + thought
                child_score = self.evaluate(new_state)
                heappush(queue, (-child_score, depth + 1, counter, new_state))
                counter += 1

        return best_state


if __name__ == "__main__":
    tot = TreeOfThoughts(branch_factor=2, max_depth=2)
    initial = "24 点游戏：4, 9, 10, 13"
    best = tot.search(initial)
    print("[最佳推理状态]")
    print(best)
