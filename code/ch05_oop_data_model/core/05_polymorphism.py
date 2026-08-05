# ---
# chapter: 5
# topic: Python 面向对象与数据模型
# topic_id: oop_data_model.polymorphism
# difficulty: ⭐⭐⭐
# tier: core
# deps: 无
# run: python 05_polymorphism.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../../../05_Python面向对象与数据模型.md
#
# Interview hooks:
# 1. 什么是鸭子类型？和 Java/C++ 的多态有什么本质区别？
# 2. 抽象基类（ABC）的意义是什么？什么场景必须用？
# 3. register() 注册虚拟子类有什么风险？

"""
多态 —— Python 采用鸭子类型（Duck Typing）

"如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子"

不需要显式继承接口，只要实现了需要的方法即可
"""

# ─────────────────────────────────────────────────────────────
# 鸭子类型演示
# ─────────────────────────────────────────────────────────────


class Dog:
    def speak(self):
        return "Woof!"


class Cat:
    def speak(self):
        return "Meow!"


class Duck:
    def speak(self):
        return "Quack!"


# 多态函数 —— 不检查类型，只检查行为
def animal_sound(animal):
    """任何有 speak() 方法的对象都可以传入"""
    return animal.speak()


# 不同类型的对象可以统一处理
for animal in [Dog(), Cat(), Duck()]:
    print(animal_sound(animal))
# Woof!
# Meow!
# Quack!

# ─────────────────────────────────────────────────────────────
# 抽象基类（ABC）— 强制接口实现
# ─────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod


class Shape(ABC):
    """形状抽象基类 —— 子类必须实现 area() 和 perimeter()"""

    @abstractmethod
    def area(self) -> float:
        """计算面积"""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """计算周长"""
        pass

    def describe(self):
        """具体方法 —— 子类可直接使用"""
        return f"面积: {self.area():.2f}, 周长: {self.perimeter():.2f}"


class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


# shape = Shape()     # TypeError: 不能实例化抽象类
rect = Rectangle(3, 4)
print(rect.describe())  # "面积: 12.00, 周长: 14.00"

# ─────────────────────────────────────────────────────────────
# 注册虚拟子类 —— 不继承但被认为是子类
# ─────────────────────────────────────────────────────────────


class Circle:
    """普通类 —— 没有显式继承 Shape"""

    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        import math

        return math.pi * self.radius**2

    def perimeter(self) -> float:
        import math

        return 2 * math.pi * self.radius


# 注册为 Shape 的虚拟子类
Shape.register(Circle)
print(issubclass(Circle, Shape))  # True
print(isinstance(Circle(1), Shape))  # True

if __name__ == "__main__":
    print("OK")
