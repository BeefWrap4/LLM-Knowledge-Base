# ---
# chapter: 2
# topic: 嵌套可变对象的引用关系
# section: 2.1.5
# difficulty: ⭐⭐
# tier: core
# deps: 无
# run: python 07_nested_refs_intro.py
# expected_runtime: <1s
# expected_output: 演示 nested[:] 是浅拷贝,创建新外层列表
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#2-1-5-内存中的对象引用关系图解
# Interview hooks:
#   1. nested[:] 创建了新对象吗?内层子列表呢?
#   2. 为什么浅拷贝对单层结构够用,对嵌套结构就不够?
#   3. 嵌套结构修改内层元素会怎样?

"""
嵌套可变对象的引用关系(深拷贝/浅拷贝的前置知识)
"""

# ─────────────────────────────────────────────────────────────
# 嵌套可变对象的引用关系(深拷贝/浅拷贝的前置知识)
# ─────────────────────────────────────────────────────────────

nested = [[1, 2], [3, 4]]
shallow = nested[:]  # 浅拷贝

if __name__ == "__main__":
    print(f"nested is shallow: {nested is shallow}")  # False
    print(f"nested[0] is shallow[0]: {nested[0] is shallow[0]}")  # True
    print("OK")
