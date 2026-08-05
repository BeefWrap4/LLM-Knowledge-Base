# ---
# chapter: 1
# topic: Python 运行时与工程环境
# topic_id: python_runtime.flow_control
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 09_flow_control.py
# expected_runtime: <1s
# expected_output: 流程控制示例
# ---
# See: ../../../01_Python运行时与工程环境.md
# Interview hooks:
#   1. for-else 结构的 else 何时执行?典型应用?
#   2. enumerate 与 zip 的用法与陷阱?
#   3. 链式比较 1 < x < 10 的等价写法?
"""
流程控制：条件与循环
"""

# ─────────────────────────────────────────────────────────────
# 条件语句 — 面试陷阱：三目运算符
# ─────────────────────────────────────────────────────────────

# Python 三目运算符（条件表达式）
# 语法：value_if_true if condition else value_if_false
age = 20
status = "成年" if age >= 18 else "未成年"  # ✅ 简洁写法
print(f"状态: {status}")

# 链式比较（Python 特色）
x = 5
print(1 < x < 10)  # True — 等价于 1 < x and x < 10
print(1 < x > 3)  # True — 可读性差，不推荐

# ─────────────────────────────────────────────────────────────
# for 循环 — 遍历序列
# ─────────────────────────────────────────────────────────────

# enumerate() — 同时获取索引和值
fruits = ["apple", "banana", "cherry"]
for idx, fruit in enumerate(fruits, start=1):  # start 参数指定起始编号
    print(f"{idx}. {fruit}")

# zip() — 并行遍历多个序列
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# zip 长度不一致时的处理
from itertools import zip_longest

a = [1, 2, 3]
b = ["a", "b"]
for x, y in zip_longest(a, b, fillvalue="N/A"):
    print(x, y)  # 1 a / 2 b / 3 N/A

# ─────────────────────────────────────────────────────────────
# break / continue / else（for-else 是面试常考点）
# ─────────────────────────────────────────────────────────────


def find_prime(n: int) -> bool:
    """
    for-else 结构：循环正常结束（未 break）时执行 else
    用于判断循环是否因 break 而中断
    """
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            print(f"{n} = {i} × {n // i}")
            break
    else:
        # 循环未 break 执行到这里 → n 是质数
        print(f"{n} 是质数")
        return True
    return False


find_prime(17)  # 17 是质数
find_prime(15)  # 15 = 3 × 5

if __name__ == "__main__":
    print("OK")
