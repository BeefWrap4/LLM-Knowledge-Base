# ---
# chapter: 29
# topic: Sub-Agent 模式 — 每个子任务拥有独立 context
# section: 29.6
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无
# run: python 09_sub_agent_pattern.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.6
# Cross-refs:
#   - Ch15 Agent (ReAct / Multi-agent)
#   - Ch18 LangGraph (supervisor 模式)
#
# Interview hooks:
#   - "Sub-Agent 模式的优缺点?"  →  优: 干净 context/并行/隔离; 缺: 协调/调试/共享状态
#   - "Sub-Agent vs 单 Agent?"   →  多步复杂任务用 Sub-Agent; 简单任务单 Agent 即可
#   - "代表实现?"                →  Claude Code / Cursor / Devin / LangGraph supervisor

from __future__ import annotations
import concurrent.futures
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubAgent:
    name: str
    system_prompt: str
    context: dict = field(default_factory=dict)   # 隔离的 context
    result: Any = None

    def run(self, task: dict) -> Any:
        # 模拟 LLM 调用: 把 task 拼到 system_prompt, 返回固定格式
        self.context["task"] = task
        # mock 输出
        if self.name == "search":
            self.result = {"hits": 3, "summary": f"已搜索 '{task.get('q','')}'"}
        elif self.name == "code":
            self.result = {"snippet": f"def solve_{task.get('q','').replace(' ', '_')}(): pass"}
        elif self.name == "data":
            self.result = {"stats": {"mean": 0.42, "std": 0.13}}
        return self.result


@dataclass
class MainAgent:
    """Supervisor: 拆解任务 -> 派发给 sub-agent -> 聚合。"""
    subs: list[SubAgent]

    def dispatch_parallel(self, tasks: dict[str, dict]) -> dict[str, Any]:
        """并行执行 sub-agent。"""
        by_name = {s.name: s for s in self.subs}
        results = {}

        def _run(name: str):
            return name, by_name[name].run(tasks[name])

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.subs)) as ex:
            futs = [ex.submit(_run, name) for name in tasks]
            for f in concurrent.futures.as_completed(futs):
                n, r = f.result()
                results[n] = r
        return results


def run_demo() -> None:
    main = MainAgent(subs=[
        SubAgent(
            name="search",
            system_prompt="你是一个网络搜索专家, 负责搜集信息。",
        ),
        SubAgent(
            name="code",
            system_prompt="你是一个 Python 工程师, 负责写代码。",
        ),
        SubAgent(
            name="data",
            system_prompt="你是一个数据分析师, 负责统计分析。",
        ),
    ])

    print("=== Sub-Agent 模式演示: 复杂任务 -> 拆解 -> 并行 ===\n")
    print("主 Agent 视角: 用户问 '分析 Python GIL 性能影响'")
    print("  拆解为 3 个子任务, 并行派发给 sub-agent\n")

    tasks = {
        "search": {"q": "Python GIL 性能影响"},
        "code":  {"q": "python_gil_bench"},
        "data":  {"q": "compute mean latency"},
    }

    results = main.dispatch_parallel(tasks)
    for name, r in results.items():
        print(f"[{name}] context_size={len(main.subs[next(i for i, s in enumerate(main.subs) if s.name==name)].context)} fields")
        print(f"   result: {r}")

    print("\n=== Sub-Agent 关键收益 ===")
    print("  - 每个 sub-agent 的 context 只装自己任务的 system_prompt, 避免污染")
    print("  - 并行执行, 节省 wall-clock 时间")
    print("  - 失败隔离: 一个 sub-agent 报错不影响主流程")
    print("  - 共享状态困难: 需要 main agent 显式聚合")


if __name__ == "__main__":
    run_demo()
    print("\nOK")
