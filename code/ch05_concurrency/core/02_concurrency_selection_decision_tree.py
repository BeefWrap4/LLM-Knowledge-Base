# ---
# chapter: 05
# topic: Python并发编程
# section: 5.1.4 CPU 密集型 vs IO 密集型任务的并发选型
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 02_concurrency_selection_decision_tree.py
# expected_runtime: <1s
# expected_output: prints the decision tree documentation
# ---
# See: ../tutorial/05_Python并发编程.md#514-cpu-密集型-vs-io-密集型任务的并发选型
# Interview hooks:
#   - CPU 密集型任务和 IO 密集型任务应该如何选择并发方案？
#   - 解释 asyncio 适合高并发的核心理由
#   - Python 3.13 nogil 模式下选型有何变化？
"""
并发选型决策树：

任务类型判断
    │
    ├── CPU 密集型? ──Yes──► Python 3.13+ nogil 可用?
    │                         │
    │                         ├── Yes ──► threading (多线程真并行)
    │                         │           （C扩展需线程安全适配）
    │                         │
    │                         └── No  ──► multiprocessing (多进程)
    │                                       或 asyncio + ProcessPoolExecutor
    │
    └── IO 密集型? ──Yes──► 并发量高(>1000)?
                              │
                              ├── Yes ──► asyncio (协程)
                              │
                              └── No  ──► threading (多线程)
"""

# 🆕 2026年补充：nogil 模式下的并发选型变化
"""
nogil 模式对并发选型的影响（2026年现状）：

┌─────────────────────────────────────────────────────────────┐
│                    nogil 模式适用性评估                        │
├─────────────────────────────────────────────────────────────┤
│  ✅ 纯 Python CPU 计算 — 多线程可直接替代多进程               │
│  ⚠️  使用 C 扩展的代码 — 需确认扩展已适配线程安全              │
│  ❌  依赖 GIL 保证线程安全的旧 C 扩展 — 暂不可用               │
│  ✅  高并发 Web 服务 — 结合 asyncio + 线程池更灵活             │
└─────────────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print("Decision tree documentation printed above.")
    print("OK")
