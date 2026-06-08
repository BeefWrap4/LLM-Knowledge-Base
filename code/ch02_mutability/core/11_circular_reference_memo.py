# ---
# chapter: 2
# topic: 深拷贝的循环引用与 memo 机制
# section: 2.2.4
# difficulty: ⭐⭐⭐
# tier: core
# deps: copy
# run: python 11_circular_reference_memo.py
# expected_runtime: <1s
# expected_output: 演示 deepcopy 通过 memo 字典处理循环引用/自引用
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#2-2-4-循环引用的处理机制
# Interview hooks:
#   1. 深拷贝遇到循环引用会无限递归吗?为什么?
#   2. memo 字典的作用是什么?两个用途?
#   3. 自引用列表(如 a = [1, 2]; a.append(a))深拷贝后,新列表还能保持自引用吗?

"""
深拷贝的循环引用处理 —— 面试进阶考点

问题:如果对象 A 引用 B,B 又引用 A,深拷贝会无限递归吗?
答案:不会。deepcopy 使用 memo 字典记录已拷贝的对象
"""

import copy

# ─────────────────────────────────────────────────────────────
# 循环引用演示
# ─────────────────────────────────────────────────────────────


def demo_circular_reference():
    """
    创建循环引用的数据结构:

    ┌──────────┐      ┌──────────┐
    │   a      │─────▶│   b      │
    │ {"x": 1} │      │ {"y": 2} │
    │          │◀─────│          │
    └──────────┘      └──────────┘
         │                 │
         └─────────────────┘
              互相引用
    """
    a = {"name": "A", "ref": None}
    b = {"name": "B", "ref": a}
    a["ref"] = b  # 建立循环引用

    print("=== 循环引用对象 ===")
    print(f"a['ref'] is b? {a['ref'] is b}")  # True
    print(f"b['ref'] is a? {b['ref'] is a}")  # True

    # 深拷贝处理循环引用
    a_copy = copy.deepcopy(a)

    print("\n深拷贝后:")
    print(f"a_copy['ref'] is b? {a_copy['ref'] is b}")  # False
    print(f"a_copy['ref']['ref'] is a_copy? {a_copy['ref']['ref'] is a_copy}")  # True
    print(f"a_copy['ref']['name']: {a_copy['ref']['name']}")  # "B"

    # 验证独立性
    a_copy["name"] = "A_copy"
    a_copy["ref"]["name"] = "B_copy"
    print("\n修改后:")
    print(f"原始 a['name']: {a['name']}")  # "A" — 不变
    print(f"原始 b['name']: {b['name']}")  # "B" — 不变


# ─────────────────────────────────────────────────────────────
# deepcopy 的 memo 机制源码级理解
# ─────────────────────────────────────────────────────────────


def deepcopy_with_memo(obj, memo=None):
    """
    模拟 deepcopy 的核心逻辑(简化版)

    memo 是一个字典:{id(原对象): 拷贝对象}
    用于:
    1. 防止循环引用导致无限递归
    2. 确保同一对象多次引用时指向同一个拷贝
    """
    if memo is None:
        memo = {}

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]  # 已拷贝过,直接返回引用

    # 创建新对象(简化版,只处理列表)
    if isinstance(obj, list):
        new_obj = []
        memo[obj_id] = new_obj  # 先放入 memo,防止循环
        for item in obj:
            new_obj.append(deepcopy_with_memo(item, memo))
        return new_obj

    # 不可变对象直接返回(无需拷贝)
    return obj


# 验证
a = [1, 2]
a.append(a)  # 自引用 [1, 2, [...]]
copied = deepcopy_with_memo(a)
print("\n自引用深拷贝:")
print(f"copied: {copied}")
print(f"copied[2] is copied? {copied[2] is copied}")  # True — 循环引用保持

if __name__ == "__main__":
    demo_circular_reference()
