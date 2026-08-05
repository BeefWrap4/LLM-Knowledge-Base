# ---
# chapter: 4
# topic: Python 迭代协议与函数式编程
# topic_id: iteration_functional.functional_styles
# difficulty: ⭐⭐⭐
# tier: core
# deps: functools, operator
# run: python 15_functional_styles.py
# expected_runtime: <1s
# expected_output: 三种风格 (imperative/functional/pythonic) 都输出 220
# ---
# See: ../../../04_Python迭代协议与函数式编程.md
# Interview hooks:
#   1. map/filter/reduce 与 Python 推导式/生成器表达式的取舍？
#   2. 什么时候命令式风格更合适？什么时候函数式更合适？
#   3. 为什么 Python 推崇"Pythonic"风格（生成器表达式）？

"""
命令式 vs 函数式风格对比
"""

# ─────────────────────────────────────────────────────────────
# 场景：计算列表中偶数的平方和
# ─────────────────────────────────────────────────────────────

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# 命令式风格（显式循环）
def imperative_sum(numbers):
    total = 0
    for n in numbers:
        if n % 2 == 0:
            total += n**2
    return total


# 函数式风格（map/filter/reduce）
def functional_sum(numbers):
    from functools import reduce
    from operator import add

    evens = filter(lambda n: n % 2 == 0, numbers)
    squares = map(lambda n: n**2, evens)
    return reduce(add, squares, 0)


# Pythonic 风格（生成器表达式 —— 推荐）
def pythonic_sum(numbers):
    return sum(n**2 for n in numbers if n % 2 == 0)


print(imperative_sum(numbers))  # 220
print(functional_sum(numbers))  # 220
print(pythonic_sum(numbers))  # 220


if __name__ == "__main__":
    print("OK")
