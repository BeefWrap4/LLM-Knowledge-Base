# ---
# chapter: 3
# topic: __getattr__ vs __getattribute__
# section: 3.5.2 __getattr__ vs __getattribute__
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 12_getattr_vs_getattribute.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/03_Python面向对象编程.md#3.5.2-__getattr__-vs-__getattribute__
#
# Interview hooks:
# 1. __getattr__ 和 __getattribute__ 的触发时机有什么不同？
# 2. 在 __getattribute__ 中用 self.x 访问属性会怎样？如何避免无限递归？
# 3. 什么场景用 __getattr__？什么场景用 __getattribute__？

"""
属性访问拦截 —— 面试高频考点

__getattr__:     仅在属性不存在时调用
__getattribute__: 任何属性访问都调用（包括存在的属性）
"""


class AttributeDemo:
    def __init__(self):
        self.existing = "我存在"

    def __getattr__(self, name):
        """属性不存在时调用 —— 可用于懒加载、动态属性"""
        print(f"__getattr__ 被调用: '{name}' 不存在")
        if name == "dynamic":
            value = f"动态创建的 {name}"
            setattr(self, name, value)  # 缓存
            return value
        raise AttributeError(f"'{type(self).__name__}' 没有 '{name}' 属性")

    def __getattribute__(self, name):
        """任何属性访问都经过这里 —— 慎用，容易无限递归"""
        print(f"__getattribute__ 被调用: '{name}'")
        # 必须用 object.__getattribute__ 获取，否则无限递归！
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        """拦截属性赋值"""
        print(f"__setattr__: {name} = {value!r}")
        super().__setattr__(name, value)


obj = AttributeDemo()
print(obj.existing)  # 先 __getattribute__，返回值
print(obj.dynamic)  # 先 __getattribute__（找不到），再 __getattr__
print(obj.dynamic)  # 第二次直接从 __getattribute__ 找到（已缓存）
# obj.nonexistent      # __getattribute__ → __getattr__ → AttributeError

"""
⚠️ __getattribute__ 使用警告：

如果在 __getattribute__ 中用 self.xxx 访问属性，
会再次触发 __getattribute__，导致无限递归！

必须用 object.__getattribute__(self, name) 来获取属性值。
"""

if __name__ == "__main__":
    print("OK")
