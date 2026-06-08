# ---
# chapter: 4
# topic: Python高级特性与函数式编程
# section: 4.2.3 带参数装饰器 —— 三层嵌套
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: functools, time
# run: python 06_decorator_retry_args.py
# expected_runtime: ~0.3s
# expected_output: 重试装饰器在第 3 次成功 — "成功！（第 3 次）"
# ---
# See: ../tutorial/04_Python高级特性与函数式编程.md — §4.2.3 带参数装饰器 —— 三层嵌套
# Interview hooks:
#   1. 为什么带参数装饰器需要三层嵌套？分别是什么角色？
#   2. 如何让重试装饰器只捕获特定异常类型？
#   3. 装饰器参数和被装饰函数参数是如何传递的？

"""
带参数的装饰器 —— 面试高频考点

需要三层嵌套：
    第一层：接收装饰器参数
    第二层：接收被装饰函数
    第三层：包装函数（实际调用）
"""

import time
from functools import wraps

# ─────────────────────────────────────────────────────────────
# 面试真题：手写重试装饰器
# ─────────────────────────────────────────────────────────────


def retry(max_attempts=3, delay=1, exceptions=(Exception,)):
    """
    重试装饰器 —— 函数失败时自动重试

    Args:
        max_attempts: 最大重试次数（含首次调用）
        delay: 每次重试间隔（秒）
        exceptions: 需要捕获的异常类型

    用法：
        @retry(max_attempts=3, delay=0.5)
        def fetch_data():
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        print(f"❌ {func.__name__} 在 {max_attempts} 次尝试后失败: {e}")
                        raise
                    print(f"⚠️ {func.__name__} 第 {attempt} 次失败: {e}，{delay}秒后重试...")
                    time.sleep(delay)
            return None  # 不会执行到这里

        return wrapper

    return decorator


# 使用重试装饰器
@retry(max_attempts=3, delay=0.1, exceptions=(ConnectionError,))
def unstable_api(call_count=[0]):
    """模拟不稳定的 API"""
    call_count[0] += 1
    if call_count[0] < 3:
        raise ConnectionError("连接超时")
    return f"成功！（第 {call_count[0]} 次）"


print(unstable_api())
# ⚠️ unstable_api 第 1 次失败: 连接超时，0.1秒后重试...
# ⚠️ unstable_api 第 2 次失败: 连接超时，0.1秒后重试...
# 成功！（第 3 次）


if __name__ == "__main__":
    print("OK")
