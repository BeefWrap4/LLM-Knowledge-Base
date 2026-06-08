# ---
# chapter: 3
# topic: __new__ vs __init__
# section: 3.4.1 核心区别
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 09_new_vs_init.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/03_Python面向对象编程.md#3.4.1-核心区别
#
# Interview hooks:
# 1. __new__ 和 __init__ 的调用顺序、参数、返回值有什么区别？
# 2. 什么场景下需要重写 __new__ 而不是 __init__？
# 3. 不可变类型（int/str/tuple）的子类化为什么要重写 __new__？

"""
__new__ vs __init__ —— 面试超高频考点

        实例化流程
    ┌─────────────────────────────────────────┐
    │                                         │
    │   1. 调用 __new__(cls, *args, **kwargs)  │
    │      ↓ 创建并返回实例对象（分配内存）      │
    │   2. 调用 __init__(self, *args, **kwargs) │
    │      ↓ 初始化实例属性                     │
    │   3. 返回初始化后的实例                    │
    │                                         │
    └─────────────────────────────────────────┘

┌─────────────┬─────────────────────────┬─────────────────────────┐
│    特性      │        __new__          │        __init__         │
├─────────────┼─────────────────────────┼─────────────────────────┤
│ 调用时机     │ 创建实例之前             │ 实例创建之后             │
│ 第一个参数   │ cls（类本身）            │ self（实例本身）         │
│ 返回值      │ 必须返回实例对象          │ 不能返回任何值（None）    │
│ 静态/实例   │ 静态方法（特殊处理）       │ 实例方法                │
│ 典型用途    │ 控制实例创建（单例等）     │ 初始化属性              │
│ 触发方式    │ 实例化时自动调用          │ __new__ 返回实例后自动调用 │
└─────────────┴─────────────────────────┴─────────────────────────┘
"""


class Demo:
    def __new__(cls, *args, **kwargs):
        print(f"1. __new__ 被调用: cls={cls.__name__}")
        instance = super().__new__(cls)  # 必须调用父类的 __new__ 创建实例
        print(f"   实例已创建: id={id(instance)}")
        return instance  # 必须返回实例

    def __init__(self, name):
        print(f"2. __init__ 被调用: self={id(self)}")
        self.name = name
        print(f"   属性已初始化: name={name}")


obj = Demo("test")
# 1. __new__ 被调用: cls=Demo
#    实例已创建: id=...
# 2. __init__ 被调用: self=...
#    属性已初始化: name=test

if __name__ == "__main__":
    print("OK")
