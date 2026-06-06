# ---
# chapter: 1
# topic: Python 3.14 类型注解改进
# section: 1.1.3
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 04_python_314_types.py
# expected_runtime: <1s
# expected_output: 类型系统示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 140-162)
# Interview hooks:
#   1. PEP 695 引入的 type 语句是什么?
#   2. Python 3.12+ 泛型类型别名的写法?
#   3. TypedDict 与 dataclass 的互操作改进?
"""
🆕 Python 3.14 类型注解改进
"""

# 1. 泛型类型别名语法 — 使用 type 语句（PEP 695 的延伸）
from typing import TypeVar

T = TypeVar('T')

# Python 3.12+ 方式
# Point = tuple[float, float]

# 3.14 支持的更清晰语法
# type Point[T] = tuple[T, T]  # 泛型类型别名

# 2. 更完善的 TypedDict 和 dataclass 互操作

# 3. 类型收窄（Type Narrowing）行为改进
#    isinstance()  narrowing 在更多场景下生效

# 4. 更好的错误信息 — 类型相关报错信息更精确

print("TypeVar 演示:", T)
print("Python 3.14 类型系统改进说明（注释文档）")

if __name__ == "__main__":
    print("OK")
