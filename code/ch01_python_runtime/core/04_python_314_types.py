# ---
# chapter: 1
# topic: Python 运行时与工程环境
# topic_id: python_runtime.python_314_types
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 04_python_314_types.py
# expected_runtime: <1s
# expected_output: 类型系统示例
# ---
# See: ../../../01_Python运行时与工程环境.md
# Interview hooks:
#   1. PEP 649/749 的延迟求值注解解决什么问题?
#   2. PEP 695 的泛型类型别名从哪个版本引入?
#   3. 读取运行时注解时为什么应使用 annotationlib?
"""
🆕 Python 3.14 类型注解改进
"""

# 1. Python 3.14 默认延迟求值注解（PEP 649 / PEP 749）。
from typing import TypeVar

T = TypeVar("T")

# 2. `type Point[T] = tuple[T, T]` 是 Python 3.12 引入的 PEP 695 语法，
#    不是 Python 3.14 新语法。为保持本示例可在 Python 3.10+ 运行，此处不直接执行。


def repeat(value: "T", count: int) -> list["T"]:
    """字符串形式也能演示注解不会阻止旧版本解析本文件。"""
    return [value] * count

print("TypeVar 演示:", T)
print("repeat 演示:", repeat("A", 2))
print("Python 3.14 类型系统改进说明（注释文档）")

if __name__ == "__main__":
    print("OK")
