# ---
# chapter: 3
# topic: 类与对象基础
# section: 3.1.1 类的定义与实例化
# difficulty: ⭐⭐⭐
# tier: core
# deps: 无
# run: python 01_class_definition.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/03_Python面向对象编程.md#3.1.1-类的定义与实例化
#
# Interview hooks:
# 1. Python 中类本身也是对象吗？是什么的实例？
# 2. 类属性和实例属性的区别？可变类属性为什么是陷阱？
# 3. @classmethod 和 @staticmethod 的区别？什么时候用哪个？

"""
类与对象 —— Python OOP 基础

Python 中一切都是对象，包括类本身（类是 type 的实例）
"""

# ─────────────────────────────────────────────────────────────
# 类定义与实例化
# ─────────────────────────────────────────────────────────────

class Dog:
    """类的文档字符串（docstring）"""

    # 类属性 —— 所有实例共享
    species = "Canis familiaris"
    count = 0

    def __init__(self, name: str, age: int):
        """
        构造方法 —— 实例化时自动调用
        self 指向新创建的实例对象
        """
        # 实例属性 —— 每个实例独立
        self.name = name
        self.age = age
        Dog.count += 1   # 修改类属性

    def bark(self):
        """实例方法 —— 第一个参数必须是 self"""
        return f"{self.name} says: Woof!"

    @classmethod
    def create_puppy(cls, name: str):
        """类方法 —— 第一个参数是 cls，可访问类属性"""
        return cls(name, age=0)   # 创建 0 岁的小狗

    @staticmethod
    def is_valid_age(age: int) -> bool:
        """静态方法 —— 无 self 或 cls，像普通函数"""
        return 0 <= age <= 30

# 实例化
dog1 = Dog("Buddy", 3)
dog2 = Dog.create_puppy("Max")

print(dog1.bark())              # "Buddy says: Woof!"
print(Dog.is_valid_age(5))      # True
print(f"共创建了 {Dog.count} 只狗")

# ─────────────────────────────────────────────────────────────
# 🎯 面试陷阱：类属性 vs 实例属性
# ─────────────────────────────────────────────────────────────

class Trap:
    # 类属性 —— 可变类型的陷阱！
    items = []   # ❌ 所有实例共享同一个列表

t1 = Trap()
t2 = Trap()
t1.items.append(1)
print(t2.items)   # [1] — t2 也被修改了！

class Safe:
    def __init__(self):
        self.items = []   # ✅ 每个实例有自己的列表

s1 = Safe()
s2 = Safe()
s1.items.append(1)
print(s2.items)   # [] — 独立

if __name__ == "__main__":
    print("OK")
