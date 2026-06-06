# ---
# chapter: 2
# topic: 浅拷贝的三种实现方式
# section: 2.2.1
# difficulty: ⭐⭐
# tier: core
# deps: copy
# run: python 08_shallow_copy_three_ways.py
# expected_runtime: <1s
# expected_output: 对比 [:] / list() / copy.copy() 三种浅拷贝方式
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#2-2-1-浅拷贝的三种实现方式
# Interview hooks:
#   1. 浅拷贝有哪三种实现方式?效果有何异同?
#   2. 浅拷贝对外层和内层的影响有何不同?
#   3. 浅拷贝修改内层元素会不会影响原对象?

"""
浅拷贝(Shallow Copy)— 创建新容器对象,但只复制最外层,
                        内部元素共享引用

三种实现方式:
1. 切片操作 [:]
2. 工厂方法 list(), dict(), set()
3. copy 模块的 copy.copy()
"""

import copy

# ─────────────────────────────────────────────────────────────
# 方式1:切片操作
# ─────────────────────────────────────────────────────────────
original = [[1, 2], [3, 4]]
shallow1 = original[:]

print(original is shallow1)           # False — 不同列表对象
print(original[0] is shallow1[0])     # True  — 子列表共享!

# 修改浅拷贝的外层(互不影响)
shallow1.append([5, 6])
print(f"original: {original}")   # [[1, 2], [3, 4]] — 不受影响
print(f"shallow1: {shallow1}")   # [[1, 2], [3, 4], [5, 6]]

# 修改浅拷贝的内层(影响原对象!)
shallow1[0].append(999)
print(f"original: {original}")   # [[1, 2, 999], [3, 4]] — 被修改了!

# ─────────────────────────────────────────────────────────────
# 方式2:工厂方法
# ─────────────────────────────────────────────────────────────
shallow2 = list(original)

# ─────────────────────────────────────────────────────────────
# 方式3:copy 模块
# ─────────────────────────────────────────────────────────────
shallow3 = copy.copy(original)

# ─────────────────────────────────────────────────────────────
# 三种方式对比
# ─────────────────────────────────────────────────────────────

def compare_shallow_methods():
    """三种浅拷贝方式的效果对比"""
    original = [[1, 2], {"a": 3}]

    methods = {
        "切片 [:]": original[:],
        "list()": list(original),
        "copy.copy()": copy.copy(original),
    }

    for name, copied in methods.items():
        print(f"\n{name}:")
        print(f"  外层对象相同? {original is copied}")
        print(f"  内层列表相同? {original[0] is copied[0]}")
        print(f"  内层字典相同? {original[1] is copied[1]}")

if __name__ == "__main__":
    compare_shallow_methods()
    print("OK")
