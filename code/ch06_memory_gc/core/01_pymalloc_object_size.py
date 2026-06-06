# ---
# chapter: Ch06
# topic: Python 内存分配器与对象大小
# section: 6.1.1 Python 内存分配器架构
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 01_pymalloc_object_size.py
# expected_runtime: < 1s
# expected_output: 打印常见对象的内存占用
# ---
# See: ../tutorial/06_Python内存管理与垃圾回收.md#61-内存管理机制
# Interview hooks:
#   1. Python 内存分配器的三层架构是什么? 各层管理多大的对象?
#   2. pymalloc 的 arena / pool / block 之间的关系?
#   3. 为什么 Python 对小对象 (<=512B) 使用 pymalloc 而不是直接 malloc?
import sys


def main() -> None:
    """演示 sys.getsizeof 的使用与对象内存布局。"""
    # 查看常见内置对象的内存大小
    print("int(42)   :", sys.getsizeof(42), "bytes")
    print("'hello'   :", sys.getsizeof("hello"), "bytes")
    print("[1,2,3]   :", sys.getsizeof([1, 2, 3]), "bytes")
    print("{'a':1}   :", sys.getsizeof({"a": 1}), "bytes")

    # pymalloc 仅在 CPython 实现中存在, sys._pymem_in_use 是 3.13+ 的 API,
    # 在更早版本上不暴露时直接跳过即可.
    pymem = getattr(sys, "_pymem_in_use", None)
    if callable(pymem):
        print("pymalloc in use:", pymem())
    else:
        print("当前 Python 版本未暴露 sys._pymem_in_use (需要 3.13+)")


if __name__ == "__main__":
    main()
    print("OK")
