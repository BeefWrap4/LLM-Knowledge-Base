# ---
# chapter: 2
# topic: 赋值 vs 浅拷贝 vs 深拷贝 完整对比
# section: 2.2.6
# difficulty: ⭐⭐
# tier: core
# deps: copy
# run: python 13_full_comparison.py
# expected_runtime: <1s
# expected_output: 表格对比 三种操作在外层/嵌套对象上是否相同
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#2-2-6-完整对比赋值-vs-浅拷贝-vs-深拷贝
# Interview hooks:
#   1. 赋值、浅拷贝、深拷贝,三者在创建新对象和拷贝嵌套对象上的区别?
#   2. 什么场景必须用深拷贝?什么场景浅拷贝就够了?
#   3. 性能差异?为什么?

"""
赋值 vs 浅拷贝 vs 深拷贝 —— 终极对比
"""

import copy


def full_comparison():
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    三种操作的本质区别                                │
    ├─────────────┬───────────────┬─────────────────┬───────────────────┤
    │    操作      │   创建新对象?   │   拷贝嵌套对象?  │   适用场景         │
    ├─────────────┼───────────────┼─────────────────┼───────────────────┤
    │ 赋值 (=)    │     否         │      否          │ 共享引用          │
    │ 浅拷贝      │     是         │      否          │ 单层结构          │
    │ 深拷贝      │     是         │      是          │ 多层嵌套结构       │
    └─────────────┴───────────────┴─────────────────┴───────────────────┘
    """

    original = [
        [1, 2, 3],
        {"key": [4, 5]},
    ]

    assigned = original  # 赋值
    shallow = copy.copy(original)  # 浅拷贝
    deep = copy.deepcopy(original)  # 深拷贝

    print("=" * 60)
    print(f"{'检查项':30s} {'=':>6s} {'shallow':>8s} {'deep':>8s}")
    print("=" * 60)

    checks = [
        ("外层对象相同", original is assigned, original is shallow, original is deep),
        (
            "子列表相同",
            original[0] is assigned[0],
            original[0] is shallow[0],
            original[0] is deep[0],
        ),
        (
            "子字典相同",
            original[1] is assigned[1],
            original[1] is shallow[1],
            original[1] is deep[1],
        ),
        (
            "字典内列表相同",
            original[1]["key"] is assigned[1]["key"],
            original[1]["key"] is shallow[1]["key"],
            original[1]["key"] is deep[1]["key"],
        ),
    ]

    for name, assigned_same, shallow_same, deep_same in checks:
        print(
            f"{name:30s} {'✓' if assigned_same else '✗':>6s} {'✓' if shallow_same else '✗':>8s} {'✓' if deep_same else '✗':>8s}"
        )

    # 修改验证独立性
    print("\n--- 修改 deep[0].append(999) 后 ---")
    deep[0].append(999)
    print(f"original[0]: {original[0]}")
    print(f"shallow[0]:  {shallow[0]}")
    print(f"deep[0]:     {deep[0]}")


if __name__ == "__main__":
    full_comparison()
    print("OK")
