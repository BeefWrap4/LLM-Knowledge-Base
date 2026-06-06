# ---
# chapter: 2
# topic: 浅拷贝对嵌套对象的行为 + [[0]*3]*3 陷阱
# section: 2.2.2
# difficulty: ⭐⭐
# tier: core
# deps: copy
# run: python 09_shallow_copy_nested_trap.py
# expected_runtime: <1s
# expected_output: 浅拷贝共享内层 + [[0]*3]*3 三行共享同一内层列表
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#2-2-2-浅拷贝对嵌套对象的行为
# Interview hooks:
#   1. 浅拷贝后修改外层元素 vs 修改内层元素,行为有什么不同?
#   2. 为什么 [[0]*3]*3 创建的不是 3x3 独立矩阵?三种正确写法?
#   3. 浅拷贝最常见的"陷阱"是什么?

"""
浅拷贝行为分析 —— 面试超高频考点

核心规则:浅拷贝只拷贝最外层容器,内部所有元素共享引用
"""

# ─────────────────────────────────────────────────────────────
# 嵌套结构浅拷贝演示
# ─────────────────────────────────────────────────────────────

def demo_shallow_copy_behavior():
    """
    ┌─────────────────────────────────────────────────────────┐
    │  原对象结构:                                             │
    │                                                         │
    │  data = [                                               │
    │      [1, 2, 3],          ← 子列表1                      │
    │      {"a": [4, 5]},      ← 子字典(内含列表)            │
    │      (6, 7)              ← 子元组(不可变)              │
    │  ]                                                      │
    │                                                         │
    │  shallow = copy.copy(data)                              │
    │                                                         │
    │  结果:data[0] is shallow[0] → True(子列表共享)        │
    │       data[1] is shallow[1] → True(子字典共享)        │
    │       data[2] is shallow[2] → True(元组共享,但不可变) │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    """
    data = [
        [1, 2, 3],
        {"a": [4, 5]},
        (6, 7)
    ]
    shallow = copy.copy(data)

    print("=== 浅拷贝后的状态 ===")
    print(f"外层相同? {data is shallow}")           # False
    print(f"子列表相同? {data[0] is shallow[0]}")    # True
    print(f"子字典相同? {data[1] is shallow[1]}")    # True
    print(f"子元组相同? {data[2] is shallow[2]}")    # True

    # 修改浅拷贝的外层 — 不影响原对象
    shallow.append("new")
    print(f"\n添加外层元素后:")
    print(f"  data: {data}")
    print(f"  shallow: {shallow}")

    # 修改浅拷贝的子列表 — 影响原对象!
    shallow[0].append(999)
    print(f"\n修改子列表后:")
    print(f"  data: {data}")           # [[1, 2, 3, 999], ...] — 变了!
    print(f"  shallow: {shallow}")

    # 替换浅拷贝的子列表 — 不影响原对象
    shallow[0] = ["new list"]
    print(f"\n替换子列表后:")
    print(f"  data: {data}")           # 不变
    print(f"  shallow: {shallow}")

# ─────────────────────────────────────────────────────────────
# 🎯 面试陷阱:多维列表的复制
# ─────────────────────────────────────────────────────────────

"""
🎯 面试题:如何创建一个 3x3 的二维列表,且每个元素是独立的?
"""

# ❌ 错误写法 — 所有行共享同一个内层列表
wrong = [[0] * 3] * 3
wrong[0][0] = 1
print(wrong)   # [[1, 0, 0], [1, 0, 0], [1, 0, 0]] — 三行都变了!

# ❌ 另一个错误写法 — 用 copy
import copy
matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
wrong2 = copy.copy(matrix)   # 浅拷贝!子列表仍然共享

# ✅ 正确写法1 — 列表推导式(每行是独立创建的)
right1 = [[0] * 3 for _ in range(3)]
right1[0][0] = 1
print(right1)  # [[1, 0, 0], [0, 0, 0], [0, 0, 0]] — 只有第一行变了

# ✅ 正确写法2 — 深拷贝
right2 = copy.deepcopy(matrix)

# ✅ 正确写法3 — NumPy(如果允许使用)
# import numpy as np
# right3 = np.zeros((3, 3), dtype=int)

if __name__ == "__main__":
    demo_shallow_copy_behavior()
    print("OK")
