# ---
# chapter: 3
# topic: 类属性、实例属性、私有属性
# section: 3.1.2 类属性、实例属性、私有属性
# difficulty: ⭐⭐⭐
# tier: core
# deps: 无
# run: python 02_attributes_and_property.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/03_Python面向对象编程.md#3.1.2-类属性实例属性私有属性
#
# Interview hooks:
# 1. Python 中 _xxx 和 __xxx 有什么区别？__xxx 是否真的私有？
# 2. @property 装饰器的作用？它的底层是什么？
# 3. 什么是 Name Mangling？什么场景下会被改写？

"""
属性访问机制 —— 面试考点
"""


class BankAccount:
    """银行账户类 —— 演示属性封装"""

    bank_name = "Python Bank"  # 类属性

    def __init__(self, owner: str, balance: float):
        self.owner = owner  # 公有属性
        self._balance = balance  # 约定：单下划线表示"内部使用"
        self.__password = "123456"  # 私有属性 —— 名称改写

    # ── property 装饰器 —— 属性访问控制 ──
    @property
    def balance(self):
        """getter —— 读取 balance 时调用"""
        return self._balance

    @balance.setter
    def balance(self, value):
        """setter —— 设置 balance 时调用"""
        if value < 0:
            raise ValueError("余额不能为负数")
        self._balance = value

    @balance.deleter
    def balance(self):
        """deleter —— 删除 balance 时调用"""
        raise AttributeError("不能删除余额属性")


# 使用
account = BankAccount("Alice", 1000)
print(account.balance)  # 1000 — 调用 getter
account.balance = 2000  # 调用 setter
# account.balance = -100    # ValueError: 余额不能为负数
# del account.balance       # AttributeError

# 私有属性的名称改写（Name Mangling）
# __password → _BankAccount__password
print(dir(account))  # 可看到 _BankAccount__password
print(account._BankAccount__password)  # "123456" — 强行访问（不推荐）

"""
名称改写机制：

class 中的 __xxx 属性会被改写为 _ClassName__xxx
目的是防止子类意外覆盖父类的私有属性

┌─────────────────────────────────────────────┐
│  类 BankAccount                              │
│  ─────────────────                           │
│  owner          → 公有，可直接访问             │
│  _balance       → 约定私有，仍可直接访问        │
│  __password     → _BankAccount__password      │
│                   名称改写，难以意外访问        │
└─────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print("OK")
