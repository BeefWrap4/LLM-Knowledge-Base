# ---
# chapter: 2
# topic: is 与 == 的区别
# section: 2.1.3
# difficulty: ⭐⭐
# tier: core
# deps: sys
# run: python 03_is_vs_equality.py
# expected_runtime: <1s
# expected_output: 演示 is 与 == 的区别,小整数缓存,字符串驻留
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#2-1-3-is-与-的本质区别
# Interview hooks:
#   1. is 和 == 有什么区别?各自比较什么?
#   2. CPython 的小整数缓存范围是?为什么?
#   3. 判断 None/True/False 应该用 is 还是 ==?为什么?
#   4. 字符串驻留(String Interning)是什么?什么情况下会失效?

"""
is 与 == 的区别 —— 面试超高频考点

==  调用 __eq__() 方法,比较值是否相等
is  比较 id(),即两个引用是否指向同一内存地址
"""

# ─────────────────────────────────────────────────────────────
# 基础对比
# ─────────────────────────────────────────────────────────────

a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True  — 值相等
print(a is b)  # False — 不同对象

# ─────────────────────────────────────────────────────────────
# 整数对象复用（实现细节，不能写进业务判断）
# ─────────────────────────────────────────────────────────────

small_a = int("100")
small_b = int("100")
print(f"small_a == small_b: {small_a == small_b}")
print(f"small_a is small_b: {small_a is small_b}（CPython 实现细节，不能依赖）")

large_a = int("1000")
large_b = int("1000")
print(f"large_a == large_b: {large_a == large_b}")
print(f"large_a is large_b: {large_a is large_b}（结果不属于 Python 语言保证）")

# ─────────────────────────────────────────────────────────────
# 字符串驻留(Interning)
# ─────────────────────────────────────────────────────────────

s1 = "hello"
s2 = "hello"
print(s1 is s2)  # 常见实现会复用编译期常量，但业务逻辑不能依赖

s3 = "hello world! python"
s4 = "hello world! python"
print(s3 is s4)  # 是否复用仍是实现细节，与字符串长度不存在可靠分界线

# 强制驻留
from sys import intern

s5 = intern("a very long string" * 100)
s6 = intern("a very long string" * 100)
print(s5 is s6)  # True — 强制驻留

# ─────────────────────────────────────────────────────────────
# 🎯 面试陷阱:空值比较
# ─────────────────────────────────────────────────────────────

# None 比较
print(None is None)  # True — None 是单例
# print(None == None)     # 也返回 True,但不规范

# 空容器比较
empty_list_a, empty_list_b = [], []
empty_dict_a, empty_dict_b = {}, {}
print(empty_list_a == empty_list_b)  # True
print(empty_list_a is empty_list_b)  # False — 两个不同的空列表
print(empty_dict_a == empty_dict_b)  # True
print(empty_dict_a is empty_dict_b)  # False

# ─────────────────────────────────────────────────────────────
# 🎯 面试真题:以下代码的输出是什么?
# ─────────────────────────────────────────────────────────────


def interview_trap():
    """
    判断输出,考察对 is 和 == 的理解
    """
    a = "hello"
    b = "hello"
    print(a is b)  # True — 字符串驻留

    c = "".join(["he", "llo"])
    print(a is c)  # False — 运行时拼接,不驻留
    print(a == c)  # True — 值相等

    d = int("256")
    e = int("256")
    print(f"256 identity: {d is e}（实现细节）")

    f = int("257")
    g = int("257")
    print(f"257 identity: {f is g}（实现细节）")


if __name__ == "__main__":
    interview_trap()
    print("OK")
