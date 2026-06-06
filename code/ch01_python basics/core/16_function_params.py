# ---
# chapter: 1
# topic: 函数参数传递机制
# section: 1.4.1
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 16_function_params.py
# expected_runtime: <1s
# expected_output: 参数传递示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 890-961)
# Interview hooks:
#   1. Python 是值传递还是引用传递?
#   2. 不可变对象和可变对象作为参数的区别?
#   3. 为什么不要用可变对象作为默认参数?
"""
Python 参数传递：传对象引用（Pass by Object Reference）

核心原则：
- 不可变对象（int, str, tuple）：函数内修改相当于重新赋值，不影响外部
- 可变对象（list, dict, set）：函数内修改会影响外部
"""

# ─────────────────────────────────────────────────────────────
# 不可变对象的参数传递
# ─────────────────────────────────────────────────────────────

def increment(x):
    """试图修改不可变整数 — 不会影外部"""
    x += 1           # 创建新的 int 对象，局部变量 x 指向新对象
    print(f"函数内 x = {x}")   # 11

a = 10
increment(a)
print(f"函数外 a = {a}")       # 10 — 不变！

# ─────────────────────────────────────────────────────────────
# 可变对象的参数传递
# ─────────────────────────────────────────────────────────────

def append_item(lst, item):
    """修改可变列表 — 会影响外部！"""
    lst.append(item)  # 原地修改列表

my_list = [1, 2, 3]
append_item(my_list, 4)
print(f"函数外 my_list = {my_list}")       # [1, 2, 3, 4] — 被修改了！

# ─────────────────────────────────────────────────────────────
# 陷阱：默认参数的延迟绑定 ⭐⭐⭐⭐⭐
# ─────────────────────────────────────────────────────────────

def add_item_bad(item, items=[]):
    """❌ 危险！默认参数在函数定义时求值，只创建一次"""
    items.append(item)
    return items

print(add_item_bad(1))   # [1]
print(add_item_bad(2))   # [1, 2] — 列表保留了上次的结果！

def add_item_good(item, items=None):
    """✅ 正确！用 None 作为哨兵值，在函数体内创建新列表"""
    if items is None:
        items = []       # 每次调用都创建新列表
    items.append(item)
    return items

print(add_item_good(1))  # [1]
print(add_item_good(2))  # [2] — 正确！

"""
默认参数陷阱的底层原理：

函数对象在定义时创建，默认参数作为函数对象的属性存储：

┌─────────────────────────────────────────┐
│  函数对象 add_item_bad                   │
│  ─────────────────────────────          │
│  __defaults__ = ([],)                   │
│           │                             │
│           ▼                             │
│        同一个列表对象（函数定义时创建）      │
│        所有调用共享这个列表！               │
└─────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print("OK")
