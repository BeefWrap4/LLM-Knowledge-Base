# ---
# chapter: 1
# topic: *args 和 **kwargs — 函数参数打包与解包
# section: 1.4.2
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 17_args_kwargs.py
# expected_runtime: <1s
# expected_output: *args / **kwargs 示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 965-1037)
# Interview hooks:
#   1. 函数参数的正确顺序是什么?
#   2. keyword-only 参数(在 * 之后)的设计目的是?
#   3. * 和 ** 在函数调用和定义时的不同含义?
"""
*args 和 **kwargs — 函数参数打包与解包
"""

# ─────────────────────────────────────────────────────────────
# 参数定义顺序（面试必考）
# ─────────────────────────────────────────────────────────────

# 正确的参数顺序：
# def func(位置参数, 默认参数, *args, 关键字-only参数, **kwargs):
#     pass

def func_demo(a, b=2, *args, c=10, **kwargs):
    """
    a:     位置参数（必填）
    b:     默认参数
    *args: 多余的位置参数 → 元组
    c:     关键字-only参数（必须用关键字传入）
    **kwargs: 多余的关键字参数 → 字典
    """
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"args = {args}")
    print(f"c = {c}")
    print(f"kwargs = {kwargs}")

func_demo(1, 3, 4, 5, c=20, d=6, e=7)
# a = 1
# b = 3
# args = (4, 5)
# c = 20
# kwargs = {'d': 6, 'e': 7}

# ─────────────────────────────────────────────────────────────
# * 和 ** 的解包用法
# ─────────────────────────────────────────────────────────────

# * 解包可迭代对象
def sum_three(a, b, c):
    return a + b + c

nums = [1, 2, 3]
print(f"解包求和: {sum_three(*nums)}")  # 6 — 等价于 sum_three(1, 2, 3)

# ** 解包字典为关键字参数
def greet(name, age):
    return f"{name} is {age} years old"

person = {"name": "Alice", "age": 25}
print(f"字典解包: {greet(**person)}")   # "Alice is 25 years old"

# 组合使用
data = [1, 2]
config = {"c": 3}
# print(sum_three(*data, **config))  # 6

# ─────────────────────────────────────────────────────────────
# 仅限关键字参数（Keyword-Only Arguments）
# ─────────────────────────────────────────────────────────────

def safe_divide(a, b, *, strict=False):
    """
    * 后的所有参数必须用关键字传入
    这种设计用于避免参数顺序错误
    """
    if strict and b == 0:
        raise ValueError("除数不能为零")
    return a / b if b != 0 else float('inf')

print(f"普通除法: {safe_divide(10, 2)}")           # 5.0
print(f"严格除法: {safe_divide(10, 0, strict=True)}")  # 必须用关键字传入 strict

if __name__ == "__main__":
    print("OK")
