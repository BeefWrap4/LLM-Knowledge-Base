# ---
# chapter: 1
# topic: Python 3.14 运行时与标准库变化
# section: 1.1.3
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 03_python_314_performance.py
# expected_runtime: <1s
# expected_output: 性能改进说明
# ---
# See: ../tutorial/01_Python编程基础.md (lines 121-136)
# Interview hooks:
#   1. Python 3.14 的 JIT 为什么必须结合构建方式讨论?
#   2. PEP 750 的模板字符串解决什么问题?
#   3. 为什么不能承诺跨工作负载的固定性能提升?
"""
Python 3.14 运行时与标准库变化
"""

# 1. PEP 750 模板字符串为库提供结构化的插值输入。
#    普通 f-string 仍按原有语义生成字符串。
name = "World"
result = f"Hello {name}!"
print(f"f-string 测试: {result}")

# 2. 实验性 JIT 与 tail-call interpreter 依赖特定 CPython 构建。

# 3. 标准库加入 interpreters 与 Zstandard 支持。

# Python 3.14 没有 `from __future__ import comptime`。

if __name__ == "__main__":
    print("OK")
