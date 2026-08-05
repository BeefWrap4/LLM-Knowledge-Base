# ---
# chapter: 7
# topic: Python 并发编程
# topic_id: concurrency.nogil_mode_detection
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 03_nogil_mode_detection.py
# expected_runtime: <1s
# expected_output: prints GIL state, then runs a small lock-protected counter demo
# ---
# See: ../../../07_Python并发编程.md
# Interview hooks:
#   - Python 3.13 的 nogil 模式是什么？它如何替代 GIL？
#   - biased reference counting 的 fast path / slow path 分别是什么？
#   - nogil 模式下还需要 Lock 吗？为什么？
"""
🆕 Python 3.13+ nogil 模式使用与检测（2026年更新）
"""

import sys
import threading

# 运行时检测 nogil 状态
if hasattr(sys, "_is_gil_enabled"):
    gil_enabled = sys._is_gil_enabled()
    print(f"GIL 状态: {'启用' if gil_enabled else '禁用（nogil 模式）'}")
else:
    print("当前 Python 版本不支持 nogil 检测（需 3.13+）")

# 检测自由线程支持
if hasattr(sys, "flags") and hasattr(sys.flags, "gil"):
    print(f"编译标志: {sys.flags}")

# nogil 模式下的线程安全编程示例
counter = 0
counter_lock = threading.Lock()  # nogil 下仍需锁保护共享状态


def increment_counter(n):
    """nogil 模式下多线程可以真正并行执行此函数"""
    global counter
    for _ in range(n):
        with counter_lock:
            counter += 1


# 在 nogil 模式下，以下代码可以利用多核 CPU
# threads = [threading.Thread(target=increment_counter, args=(100000,))
#            for _ in range(4)]
# for t in threads: t.start()
# for t in threads: t.join()
# print(f"最终计数: {counter}")

# 小规模演示锁的作用
threads = [threading.Thread(target=increment_counter, args=(1000,)) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"最终计数: {counter}")  # 期望 2000

if __name__ == "__main__":
    print("OK")
