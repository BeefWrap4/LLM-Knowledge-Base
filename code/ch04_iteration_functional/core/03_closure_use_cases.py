# ---
# chapter: 3
# topic: Python 函数、作用域与装饰器
# topic_id: iteration_functional.closure_use_cases
# difficulty: ⭐⭐⭐
# tier: core
# deps: 无
# run: python 03_closure_use_cases.py
# expected_runtime: <1s
# expected_output: 函数工厂 + 闭包私有化 vs 类的对比
# ---
# See: ../../../03_Python函数作用域与装饰器.md
# Interview hooks:
#   1. 闭包相对于类实现计数器有什么优势？
#   2. 什么是函数工厂？典型应用场景是什么？
#   3. 闭包实现的数据隐藏为什么比类更"私有"？

"""
闭包的实际应用 —— 理解闭包的实用价值
"""


# 1. 函数工厂 —— 根据配置创建不同的函数
def power_factory(n):
    """创建 x^n 的函数"""

    def power(x):
        return x**n

    return power


square = power_factory(2)
cube = power_factory(3)
print(square(4))  # 16
print(cube(3))  # 27


# 2. 私有化 —— 数据隐藏
class Counter:
    """类实现"""

    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.count


def make_counter():
    """闭包实现 —— count 真正私有"""
    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    return counter


# 闭包版本：count 无法从外部访问（没有 self.count）
c = make_counter()
print(c())  # 1
print(c())  # 2
# 无法访问 count 变量！真正的私有


if __name__ == "__main__":
    print("OK")
