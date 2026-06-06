# ---
# chapter: 1
# topic: 基础数据类型与内存行为
# section: 1.2.1
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 07_basic_data_types.py
# expected_runtime: <1s
# expected_output: 各种类型示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 246-302)
# Interview hooks:
#   1. 小整数缓存范围是什么?为什么设计成 -5~256?
#   2. 浮点数 0.1+0.2 != 0.3 的根本原因?
#   3. bool 是什么的子类?True+True 的结果?
"""
Python 基础数据类型与内存行为
"""

# ─────────────────────────────────────────────────────────────
# 不可变类型（Immutable）— 创建后不可修改，修改会创建新对象
# ─────────────────────────────────────────────────────────────

# int — 任意精度整数（无溢出限制）
a = 10           # 小整数被缓存（-5 ~ 256）
b = 10
print(a is b)    # True — 缓存复用

c = 1000
d = 1000
print(c is d)    # False — 大整数不缓存

# float — 双精度浮点数（IEEE 754，64位）
pi = 3.14159
print(f"float 精度: {pi:.15f}")  # 约 15-17 位有效数字

# 浮点数精度问题（面试常考陷阱）
print(0.1 + 0.2 == 0.3)         # False！
print(f"0.1 + 0.2 = {0.1 + 0.2:.17f}")  # 0.30000000000000004
# 正确做法：使用 decimal 模块或允许误差
import math
print(math.isclose(0.1 + 0.2, 0.3))  # True

# str — Unicode 字符串，不可变
s = "hello"
# s[0] = "H"  # TypeError: 'str' object does not support item assignment
s = "H" + s[1:]  # 合法：创建新字符串
print(f"新字符串: {s}")

# bool — True/False，是 int 的子类
print(True + True)   # 2
print(isinstance(True, int))  # True

# None — 空值，单例对象
print(type(None))    # <class 'NoneType'>

# ─────────────────────────────────────────────────────────────
# 可变类型（Mutable）— 创建后可原地修改
# ─────────────────────────────────────────────────────────────

# list — 动态数组（底层是过度分配的数组）
lst = [1, 2, 3]
lst.append(4)        # 原地修改，id(lst) 不变
print(f"列表: {lst}")

# dict — 哈希表（Python 3.7+ 保持插入顺序）
d = {"a": 1, "b": 2}
d["c"] = 3           # 原地修改
print(f"字典: {d}")

# set — 哈希集合，无序不重复
se = {1, 2, 3}
se.add(4)            # 原地修改
print(f"集合: {se}")

if __name__ == "__main__":
    print("OK")
