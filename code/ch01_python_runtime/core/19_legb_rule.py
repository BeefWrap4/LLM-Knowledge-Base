# ---
# chapter: 3
# topic: Python 函数、作用域与装饰器
# topic_id: python_runtime.legb_rule
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 19_legb_rule.py
# expected_runtime: <1s
# expected_output: LEGB 规则示例
# ---
# See: ../../../03_Python函数作用域与装饰器.md
# Interview hooks:
#   1. LEGB 四个作用域的查找顺序?
#   2. global 与 nonlocal 的使用场景区别?
#   3. lambda 在循环中延迟绑定的陷阱与修复?
"""
LEGB 规则：变量查找的优先级顺序

L — Local（局部作用域）：当前函数内部
E — Enclosing（嵌套作用域）：外层嵌套函数
G — Global（全局作用域）：模块级别
B — Built-in（内置作用域）：builtins 模块
"""

# ─────────────────────────────────────────────────────────────
# LEGB 规则演示
# ─────────────────────────────────────────────────────────────

x = "global"  # G — 全局


def outer():
    x = "enclosing"  # E — 外层函数的局部变量

    def inner():
        x = "local"  # L — 本函数的局部变量
        print(x)  # "local" — 按 LEGB 找到 Local

    inner()


outer()

# ─────────────────────────────────────────────────────────────
# global 和 nonlocal 关键字
# ─────────────────────────────────────────────────────────────

counter = 0


def increment_global():
    """使用 global 修改全局变量"""
    global counter
    counter += 1


increment_global()
print(f"counter = {counter}")  # 1


def outer_counter():
    """使用 nonlocal 修改外层变量"""
    count = 0

    def inner():
        nonlocal count  # 声明使用外层（非全局）变量
        count += 1
        return count

    return inner


increment = outer_counter()
print(f"first call: {increment()}")  # 1
print(f"second call: {increment()}")  # 2
print(f"third call: {increment()}")  # 3

# ─────────────────────────────────────────────────────────────
# 🎯 面试陷阱：LEGB 与默认值捕获
# ─────────────────────────────────────────────────────────────

# 陷阱：lambda 在循环中延迟绑定
funcs = []
for i in range(3):
    funcs.append(lambda: i)  # i 是自由变量，不是默认值

print(f"延迟绑定结果: {[f() for f in funcs]}")  # [2, 2, 2] — 不是 [0, 1, 2]！

# 正确做法：用默认参数捕获当前值
funcs_correct = []
for i in range(3):
    funcs_correct.append(lambda x=i: x)  # x=i 在定义时求值

print(f"默认参数修复: {[f() for f in funcs_correct]}")  # [0, 1, 2] ✅

if __name__ == "__main__":
    print("OK")
