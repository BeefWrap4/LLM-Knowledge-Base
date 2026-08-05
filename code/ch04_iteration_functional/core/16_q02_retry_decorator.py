# ---
# chapter: 3
# topic: Python 函数、作用域与装饰器
# topic_id: iteration_functional.q02_retry_decorator
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: functools, time
# run: python 16_q02_retry_decorator.py
# expected_runtime: <1s
# expected_output: 装饰器定义演示（无 I/O 执行）
# ---
# See: ../../../03_Python函数作用域与装饰器.md
# Interview hooks:
#   1. 重试装饰器如何控制最大重试次数？
#   2. 为什么要用 functools.wraps？
#   3. 如何让重试在达到上限后重新抛出异常？

"""
Q2：手写一个带参数的装饰器（如重试装饰器）。

带参数装饰器需要三层嵌套：
第一层接收装饰器参数，第二层接收被装饰函数，第三层是包装函数。
关键点：最内层可以访问外层参数形成闭包。
"""

import time
from functools import wraps


def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)

        return wrapper

    return decorator


if __name__ == "__main__":
    # 简单冒烟测试：装饰器工厂返回 callable
    deco = retry(max_attempts=2, delay=0)
    assert callable(deco)
    print("OK")
