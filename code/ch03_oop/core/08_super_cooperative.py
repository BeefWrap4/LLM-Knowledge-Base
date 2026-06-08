# ---
# chapter: 3
# topic: super() 在多继承中的行为
# section: 3.3.3 super() 在多继承中的行为
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 08_super_cooperative.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/03_Python面向对象编程.md#3.3.3-super-在多继承中的行为
#
# Interview hooks:
# 1. 为什么多继承中所有类的 __init__ 都应该调用 super()？
# 2. 如果某个中间类忘记调用 super()，MRO 链会怎样？
# 3. 协同多继承（Cooperative Multiple Inheritance）的设计模式？

"""
super() 在多继承中的协同调用（Cooperative Multiple Inheritance）

核心设计模式：所有类都调用 super()，确保 MRO 链完整执行
"""


class Base:
    def __init__(self):
        print("Base.__init__")
        super().__init__()  # 关键：即使是基类也要调用 super()


class FeatureA(Base):
    def __init__(self):
        print("FeatureA.__init__")
        super().__init__()


class FeatureB(Base):
    def __init__(self):
        print("FeatureB.__init__")
        super().__init__()


class MyClass(FeatureA, FeatureB):
    def __init__(self):
        print("MyClass.__init__")
        super().__init__()


# MRO: MyClass -> FeatureA -> FeatureB -> Base -> object
obj = MyClass()
# MyClass.__init__
# FeatureA.__init__
# FeatureB.__init__
# Base.__init__

# ─────────────────────────────────────────────────────────────
# 🎯 面试陷阱：忘记调用 super()
# ─────────────────────────────────────────────────────────────


class BadA:
    def __init__(self):
        print("BadA.__init__")
        # ❌ 没有调用 super()


class BadB:
    def __init__(self):
        print("BadB.__init__")
        super().__init__()


class BadChild(BadA, BadB):
    def __init__(self):
        print("BadChild.__init__")
        super().__init__()


# MRO: BadChild -> BadA -> BadB -> object
BadChild()
# BadChild.__init__
# BadA.__init__
# BadB.__init__ 不会被执行！❌

if __name__ == "__main__":
    print("OK")
