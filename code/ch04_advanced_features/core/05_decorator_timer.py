# ---
# chapter: 4
# topic: Python高级特性与函数式编程
# section: 4.2.2 无参数装饰器 —— 计时器
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: functools, time
# run: python 05_decorator_timer.py
# expected_runtime: ~1s
# expected_output: 计时器输出 + 统计信息 dict
# ---
# See: ../tutorial/04_Python高级特性与函数式编程.md — §4.2.2 无参数装饰器 —— 计时器
# Interview hooks:
#   1. 为什么用 time.perf_counter() 而不是 time.time()？
#   2. 如何在装饰器上保存状态（如调用次数）？
#   3. 装饰器栈上的属性（如 wrapper.call_count）有什么作用？

"""
手写计时装饰器 —— 面试高频手撕代码题
"""

import time
from functools import wraps

def timer(func):
    """
    计时装饰器 —— 测量函数执行时间

    用法：
        @timer
        def slow_func():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()   # 高精度计时
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"⏱ {func.__name__} 执行时间: {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_sum(n):
    """计算 1+2+...+n"""
    total = 0
    for i in range(1, n + 1):
        total += i
        time.sleep(0.001)   # 模拟耗时操作
    return total

result = slow_sum(100)
print(f"结果: {result}")

# ─────────────────────────────────────────────────────────────
# 带统计功能的计时装饰器
# ─────────────────────────────────────────────────────────────

def timer_with_stats(func):
    """
    增强版计时装饰器 —— 记录调用次数和累计时间
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        wrapper.total_time += elapsed
        print(f"⏱ {func.__name__} #{wrapper.call_count}: {elapsed:.4f}s "
              f"(累计: {wrapper.total_time:.4f}s)")
        return result

    # 在 wrapper 上附加统计属性
    wrapper.call_count = 0
    wrapper.total_time = 0.0
    wrapper.get_stats = lambda: {
        "calls": wrapper.call_count,
        "total_time": wrapper.total_time,
        "avg_time": wrapper.total_time / wrapper.call_count if wrapper.call_count else 0,
    }

    return wrapper

@timer_with_stats
def compute(x):
    time.sleep(0.01)
    return x * x

compute(1)
compute(2)
compute(3)
print(f"统计: {compute.get_stats()}")


if __name__ == "__main__":
    print("OK")
