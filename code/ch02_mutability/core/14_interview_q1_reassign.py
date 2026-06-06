# ---
# chapter: 2
# topic: 面试 Q1 — 重新赋值 vs 原地修改
# section: Q1
# difficulty: ⭐⭐
# tier: core
# deps: 无
# run: python 14_interview_q1_reassign.py
# expected_runtime: <1s
# expected_output: 打印 [1, 2, 3](b 仍指向原列表)
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#q1以下代码的输出是什么为什么
# Interview hooks:
#   1. a = [1,2,3]; b = a; a = [4,5,6]; print(b) 输出什么?为什么?
#   2. 这与 a[:] = [4, 5, 6] 的行为有什么区别?
#   3. "重新赋值"与"原地修改"的本质区别?

"""
面试真题 Q1:
    a = [1, 2, 3]
    b = a
    a = [4, 5, 6]
    print(b)
答案:[1, 2, 3] — b 仍指向原列表。
"""

a = [1, 2, 3]
b = a
a = [4, 5, 6]
print(b)  # [1, 2, 3]

if __name__ == "__main__":
    print("OK")
