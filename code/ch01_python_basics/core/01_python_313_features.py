# ---
# chapter: 1
# topic: Python 3.13 核心新特性速览
# section: 1.1.2
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 01_python_313_features.py
# expected_runtime: <1s
# expected_output: GIL 状态 + 弃用警告示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 45-73)
# Interview hooks:
#   1. Python 3.13 的 no-GIL 模式是什么?如何启用?
#   2. @deprecated 装饰器的作用和使用场景?
#   3. PEP 703 的核心实现原理(biased reference counting)?
"""
Python 3.13 核心新特性速览
（以下代码需在 Python 3.13+ 环境中运行）
"""

# 1. 实验性 no-GIL 模式（自由线程）— PEP 703
#    2026年状态：已可通过官方实验性构建体验，生产环境尚不建议
#    编译时启用：--disable-gil
#    运行时检测：
import sys
import warnings

if hasattr(sys, "_is_gil_enabled"):
    print(f"GIL 状态: {sys._is_gil_enabled()}")  # True/False
else:
    print("当前 Python 版本不支持 _is_gil_enabled() (需要 3.13+)")

# 2. 改进的交互式解释器（彩色高亮、多行编辑）

# 3. 实验性 JIT 编译器（性能提升 2-9%，基于复制解释的即时编译）

# 4. 新的类型标注语法（PEP 702 警告废弃）
try:
    from warnings import deprecated

    @deprecated("请使用 new_func() 替代")
    def old_func():
        return "deprecated"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_func()
        if w:
            print(f"弃用警告触发: {w[-1].message}")
        print(f"old_func() 返回: {result}")
except ImportError:
    # Python < 3.13
    def old_func():
        return "deprecated"

    print(f"old_func() 返回: {old_func()} (Python < 3.13)")

# 5. iOS 和 Android 官方支持（移动端 Python）

# 6. os.register_at_fork() 的清理机制改进

if __name__ == "__main__":
    print("OK")
