# ---
# chapter: 3
# topic: Python 函数、作用域与装饰器
# topic_id: iteration_functional.closure_legb_nonlocal
# difficulty: ⭐⭐⭐
# tier: core
# deps: 无
# run: python 02_closure_legb_nonlocal.py
# expected_runtime: <1s
# expected_output: nonlocal 计数器 + 循环闭包陷阱演示
# ---
# See: ../../../03_Python函数作用域与装饰器.md
# Interview hooks:
#   1. LEGB 四个字母分别代表什么？
#   2. nonlocal 关键字的作用是什么？与 global 有什么区别？
#   3. 循环中创建闭包为什么会有陷阱？如何修复？

"""
LEGB 规则在闭包中的应用

L — Local:        当前函数 inner 的局部变量
E — Enclosing:    外层函数 outer 的局部变量
G — Global:       模块级别的全局变量
B — Built-in:     Python 内置变量
"""

x = "global"  # G


def outer():
    x = "enclosing"  # E

    def inner():
        x = "local"  # L
        print(x)  # "local" — 按 LEGB 找到 Local

    inner()


outer()

# ─────────────────────────────────────────────────────────────
# nonlocal —— 修改外层变量（闭包关键）
# ─────────────────────────────────────────────────────────────


def counter_factory():
    """
    闭包实现计数器 —— nonlocal 修改外层变量
    """
    count = 0  # 外层变量

    def counter():
        nonlocal count  # 声明：我要修改外层变量，不是创建局部变量
        count += 1
        return count

    def reset():
        nonlocal count
        count = 0

    # 返回多个闭包函数
    return counter, reset


cnt, reset = counter_factory()
print(cnt())  # 1
print(cnt())  # 2
print(cnt())  # 3
reset()
print(cnt())  # 1

# ─────────────────────────────────────────────────────────────
# 循环中创建闭包陷阱
# ─────────────────────────────────────────────────────────────


# 陷阱：所有闭包共享同一个循环变量
def create_functions_trap():
    """❌ 错误：所有函数都返回 4"""
    functions = []
    for i in range(4):
        functions.append(lambda: i)  # i 是自由变量，不是默认值
    return functions


funcs = create_functions_trap()
print([f() for f in funcs])  # [3, 3, 3, 3] — 不是 [0, 1, 2, 3]！


# 修复：用默认参数在定义时捕获值
def create_functions_fixed():
    """✅ 正确：每个闭包捕获当前的 i 值"""
    functions = []
    for i in range(4):
        functions.append(lambda x=i: x)  # x=i 在定义时求值
    return functions


funcs = create_functions_fixed()
print([f() for f in funcs])  # [0, 1, 2, 3] ✅


# 另一种修复：用工厂函数创建闭包
def make_closure(x):
    def closure():
        return x

    return closure


def create_functions_factory():
    return [make_closure(i) for i in range(4)]


funcs = create_functions_factory()
print([f() for f in funcs])  # [0, 1, 2, 3] ✅


if __name__ == "__main__":
    print("OK")
