# ---
# chapter: 2
# topic: 深拷贝的原理与实现
# section: 2.2.3
# difficulty: ⭐⭐
# tier: core
# deps: copy
# run: python 10_deep_copy_basics.py
# expected_runtime: <1s
# expected_output: 演示 copy.deepcopy 递归拷贝所有嵌套对象
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#2-2-3-深拷贝的原理与实现
# Interview hooks:
#   1. 深拷贝和浅拷贝的本质区别?
#   2. 深拷贝对不可变对象(如纯元素 tuple)会怎么处理?
#   3. 深拷贝为什么比浅拷贝慢?哪些情况会显著变慢?

"""
深拷贝(Deep Copy)— 递归拷贝所有层级,完全独立的对象

实现:copy.deepcopy()
原理:递归遍历对象图,创建每个对象的新副本
"""

import copy

# ─────────────────────────────────────────────────────────────
# 深拷贝基础演示
# ─────────────────────────────────────────────────────────────


def demo_deep_copy():
    original = [[1, 2, 3], {"a": [4, 5]}, (6, 7)]
    deep = copy.deepcopy(original)

    print("=== 深拷贝后的状态 ===")
    print(f"外层相同? {original is deep}")  # False
    print(f"子列表相同? {original[0] is deep[0]}")  # False — 深拷贝创建了新的!
    print(f"子字典相同? {original[1] is deep[1]}")  # False
    print(f"子元组相同? {original[2] is deep[2]}")  # True — 元组不可变,不需要拷贝
    print(f"字典内列表相同? {original[1]['a'] is deep[1]['a']}")  # False

    # 任意修改深拷贝,都不影响原对象
    deep[0].append(999)
    deep[1]["a"].append(888)
    deep[1]["b"] = "new"

    print("\n修改后:")
    print(f"  original: {original}")  # 完全不变
    print(f"  deep: {deep}")


if __name__ == "__main__":
    demo_deep_copy()
