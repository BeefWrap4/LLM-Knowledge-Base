# ---
# chapter: 2
# topic: Python 对象模型与可变性
# topic_id: object_model.object_identity_type_value
# difficulty: ⭐⭐
# tier: core
# deps: 无
# run: python 01_object_identity_type_value.py
# expected_runtime: <1s
# expected_output: 演示对象的身份(id)、类型(type)、值
# ---
# See: ../../../02_Python对象模型与可变性.md
# Interview hooks:
#   1. 什么是对象的身份、类型、值?如何获取?
#   2. Python 中变量是"盒子"还是"标签"?

"""
对象的身份、类型、值 —— Python 对象模型基础
"""

a = [1, 2, 3]
print(f"身份 (id) : {id(a)}       # 内存地址")  # 如 140234567890
print(f"类型      : {type(a)}     # <class 'list'>")
print(f"值        : {a}           # [1, 2, 3]")

# 可变 vs 不可变的本质区别
"""
┌─────────────────────────────────────────────────────────────┐
│                    可变性与不可变性的本质                      │
│                                                             │
│   不可变对象 (Immutable)          可变对象 (Mutable)          │
│   ─────────────────────           ─────────────────         │
│   • 创建后内容不可修改              • 创建后内容可以修改        │
│   • 修改操作 = 创建新对象           • 修改操作 = 原地修改       │
│   • 可作为字典键                   • 不可作为字典键            │
│   • 线程安全(无竞态条件)           • 需要同步机制保证线程安全   │
│                                                             │
│   示例: int, float, str,           示例: list, dict, set,     │
│         bool, tuple, frozenset           bytearray           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print("OK")
