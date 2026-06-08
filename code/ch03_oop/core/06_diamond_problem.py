# ---
# chapter: 3
# topic: 钻石问题（Diamond Problem）
# section: 3.3.1 钻石问题
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 06_diamond_problem.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/03_Python面向对象编程.md#3.3.1-钻石问题Diamond-Problem
#
# Interview hooks:
# 1. 什么是钻石问题？Python 如何保证基类方法只被调用一次？
# 2. 如果所有类都调用 super()，A 的 __init__ 会被执行几次？
# 3. C3 线性化和深度优先、广度优先算法有什么区别？

"""
多继承的钻石问题 —— 为什么需要 C3 线性化

    A
   / \
  B   C
   \\ /
    D

D 继承 B 和 C，B 和 C 都继承 A。
当 D 调用 super() 时，A 的方法会被调用几次？
Python 的 C3 线性化确保 A 只被调用一次！
"""


class A:
    def method(self):
        print("A.method")


class B(A):
    def method(self):
        print("B.method")
        super().method()  # 调用 MRO 中的下一个


class C(A):
    def method(self):
        print("C.method")
        super().method()


class D(B, C):
    def method(self):
        print("D.method")
        super().method()


# MRO: D -> B -> C -> A -> object
print("MRO:", [c.__name__ for c in D.__mro__])
# ['D', 'B', 'C', 'A', 'object']

D().method()
# D.method
# B.method
# C.method
# A.method
# A 只被调用一次！✅

if __name__ == "__main__":
    print("OK")
