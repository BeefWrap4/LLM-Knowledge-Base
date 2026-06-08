# ---
# chapter: 3
# topic: 继承（Inheritance）
# section: 3.2.2 继承
# difficulty: ⭐⭐⭐
# tier: core
# deps: 无
# run: python 04_inheritance.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/03_Python面向对象编程.md#3.2.2-继承Inheritance
#
# Interview hooks:
# 1. super() 真的是调用父类吗？它在多继承中调用的是谁？
# 2. 单继承和多继承中 super() 的行为差异？
# 3. 为什么每个 __init__ 都应该调用 super().__init__()？

"""
继承 —— 复用父类的属性和方法

Python 支持多继承，方法解析顺序（MRO）通过 C3 线性化算法确定
"""

# ─────────────────────────────────────────────────────────────
# 单继承
# ─────────────────────────────────────────────────────────────


class Animal:
    """动物基类"""

    def __init__(self, name: str):
        self.name = name

    def speak(self):
        raise NotImplementedError("子类必须实现此方法")

    def introduce(self):
        return f"我是 {self.name}"


class Dog(Animal):
    """狗 —— 继承 Animal"""

    def __init__(self, name: str, breed: str):
        super().__init__(name)  # 调用父类构造方法
        self.breed = breed  # 子类特有属性

    def speak(self):
        return f"{self.name}: Woof!"

    def fetch(self):
        """子类特有方法"""
        return f"{self.name} 去捡球了"


# ─────────────────────────────────────────────────────────────
# super() 的深入理解
# ─────────────────────────────────────────────────────────────

"""
super() 不是调用父类，而是按照 MRO 顺序调用下一个类！

在单继承中：super() ≈ 调用父类
在多继承中：super() 调用 MRO 列表中的下一个类

这是理解多继承的关键！
"""


class A:
    def __init__(self):
        print("A.__init__")
        super().__init__()  # 按 MRO 继续调用


class B(A):
    def __init__(self):
        print("B.__init__")
        super().__init__()  # 不是直接调用 A！是按 MRO 调用下一个


class C(B):
    def __init__(self):
        print("C.__init__")
        super().__init__()


# MRO: C -> B -> A -> object
C()
# 输出:
# C.__init__
# B.__init__
# A.__init__

if __name__ == "__main__":
    print("OK")
