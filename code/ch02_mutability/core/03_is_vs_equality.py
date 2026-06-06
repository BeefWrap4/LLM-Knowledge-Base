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
print(a == b)     # True  — 值相等
print(a is b)     # False — 不同对象

# ─────────────────────────────────────────────────────────────
# 小整数缓存(-5 ~ 256)
# ─────────────────────────────────────────────────────────────

a = 100
b = 100
print(a is b)     # True — 小整数被缓存复用

c = 1000
d = 1000
print(c is d)     # False — 大整数不缓存(交互模式下可能缓存)

# ─────────────────────────────────────────────────────────────
# 字符串驻留(Interning)
# ─────────────────────────────────────────────────────────────

s1 = "hello"
s2 = "hello"
print(s1 is s2)   # True — 编译期常量字符串被驻留

s3 = "hello world! python"
s4 = "hello world! python"
print(s3 is s4)   # 可能 False — 长字符串不保证驻留

# 强制驻留
from sys import intern
s5 = intern("a very long string" * 100)
s6 = intern("a very long string" * 100)
print(s5 is s6)   # True — 强制驻留

# ─────────────────────────────────────────────────────────────
# 🎯 面试陷阱:空值比较
# ─────────────────────────────────────────────────────────────

# None 比较
print(None is None)       # True — None 是单例
# print(None == None)     # 也返回 True,但不规范

# 空容器比较
print([] == [])           # True
print([] is [])           # False — 两个不同的空列表
print({} == {})           # True
print({} is {})           # False

# ─────────────────────────────────────────────────────────────
# 🎯 面试真题:以下代码的输出是什么?
# ─────────────────────────────────────────────────────────────

def interview_trap():
    """
    判断输出,考察对 is 和 == 的理解
    """
    a = "hello"
    b = "hello"
    print(a is b)          # True — 字符串驻留

    c = "".join(["he", "llo"])
    print(a is c)          # False — 运行时拼接,不驻留
    print(a == c)          # True — 值相等

    d = 256
    e = 256
    print(d is e)          # True — -5~256 缓存

    f = 257
    g = 257
    print(f is g)          # False(通常)— 超出缓存范围

if __name__ == "__main__":
    interview_trap()
    print("OK")
