# ---
# chapter: 5
# topic: Python 面向对象与数据模型
# topic_id: oop_data_model.descriptor_protocol
# difficulty: ⭐⭐⭐
# tier: core
# deps: 无
# run: python 13_descriptor_protocol.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../../../05_Python面向对象与数据模型.md
#
# Interview hooks:
# 1. 什么是描述符？数据描述符和非数据描述符的优先级？
# 2. 手写一个 @property —— 描述符协议的三个方法签名是什么？
# 3. ORM 框架中字段类型校验的描述符如何设计？

"""
描述符（Descriptor）— Python 属性访问的底层机制

描述符：实现了 __get__、__set__ 或 __delete__ 中至少一个的类

应用：@property、classmethod、staticmethod、ORM 字段

┌─────────────────────────────────────────────────────────────┐
│                     描述符分类                               │
│                                                             │
│   数据描述符（Data Descriptor）                              │
│   ├── 同时定义 __get__ + __set__                             │
│   └── 优先级高于实例字典                                     │
│                                                             │
│   非数据描述符（Non-data Descriptor）                         │
│   ├── 只定义 __get__                                         │
│   └── 优先级低于实例字典                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""

# ─────────────────────────────────────────────────────────────
# 手写 @property —— 理解描述符本质
# ─────────────────────────────────────────────────────────────


class MyProperty:
    """
    模拟 property 的实现 —— 数据描述符
    """

    def __init__(self, fget=None, fset=None, fdel=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel

    def __get__(self, instance, owner):
        """instance: 实例对象; owner: 类"""
        if instance is None:  # 类属性访问：Celsius.fahrenheit
            return self
        if self.fget is None:
            raise AttributeError("不可读")
        return self.fget(instance)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError("不可写")
        self.fset(instance, value)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError("不可删除")
        self.fdel(instance)

    # 支持装饰器语法
    def getter(self, fget):
        self.fget = fget
        return self

    def setter(self, fset):
        self.fset = fset
        return self


# 使用手写的 property
class Celsius:
    def __init__(self, celsius=0):
        self._celsius = celsius

    @MyProperty  # 等价于 fahrenheit = MyProperty(fget)
    def fahrenheit(self):
        return self._celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5 / 9


c = Celsius(25)
print(c.fahrenheit)  # 77.0

# ─────────────────────────────────────────────────────────────
# 类型检查描述符 —— 实用示例
# ─────────────────────────────────────────────────────────────


class Typed:
    """类型检查描述符 —— ORM 风格的字段定义"""

    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"'{self.name}' 期望 {self.expected_type.__name__}，收到 {type(value).__name__}")
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class Person:
    """使用描述符实现类型检查"""

    name = Typed("name", str)
    age = Typed("age", int)
    height = Typed("height", float)

    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height


p = Person("Alice", 25, 1.65)
# p.age = "25"  # TypeError: 'age' 期望 int，收到 str

if __name__ == "__main__":
    print("OK")
