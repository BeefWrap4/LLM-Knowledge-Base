# ---
# chapter: 2
# topic: 面试 Q6 — copy.copy vs copy.deepcopy
# section: Q6
# difficulty: ⭐⭐
# tier: core
# deps: copy
# run: python 15_interview_q6_copy_vs_deepcopy.py
# expected_runtime: <1s
# expected_output: b 包含 [2,3,4] 共享内层;c 完全独立
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#q6以下代码会输出什么
# Interview hooks:
#   1. b 和 c 在 a[1].append(4) 后分别输出什么?为什么?
#   2. 为什么浅拷贝"看起来"包含了原列表的修改?
#   3. 实战中什么时候用浅拷贝 vs 深拷贝?

"""
面试真题 Q6:
    import copy
    a = [1, [2, 3]]
    b = copy.copy(a)
    c = copy.deepcopy(a)
    a[1].append(4)
    print(b)  # ?
    print(c)  # ?
答案:
    b = [1, [2, 3, 4]]  — 浅拷贝共享内层列表
    c = [1, [2, 3]]     — 深拷贝完全独立
"""

import copy
a = [1, [2, 3]]
b = copy.copy(a)
c = copy.deepcopy(a)
a[1].append(4)
print(b)  # [1, [2, 3, 4]]
print(c)  # [1, [2, 3]]

if __name__ == "__main__":
    print("OK")
