# ---
# chapter: 4
# topic: Python高级特性与函数式编程
# section: 面试真题 Q9 —— 闭包陷阱
# difficulty: ⭐⭐⭐
# tier: core
# deps: 无
# run: python 17_q09_closure_trap.py
# expected_runtime: <1s
# expected_output: 错误版本输出 [8,8,8,8,8]，修复版本输出 [0,2,4,6,8]
# ---
# See: ../tutorial/04_Python高级特性与函数式编程.md — 面试真题 Q9
# Interview hooks:
#   1. 为什么输出 8 8 8 8 8 而不是 0 2 4 6 8？
#   2. lambda x, i=i: i * x 中 i=i 起到了什么作用？
#   3. 默认参数的值是在什么时候求值的？定义时还是调用时？

"""
Q9：以下代码的输出是什么？

    def make_multipliers():
        return [lambda x: i * x for i in range(5)]

    for m in make_multipliers():
        print(m(2), end=" ")

输出 8 8 8 8 8（不是 0 2 4 6 8）。列表推导式中的 lambda 形成了闭包，
引用了自由变量 i。当 lambda 被调用时，循环已经结束，
i 的最终值是 4。所有 lambda 都引用同一个 i，所以都返回 4 * 2 = 8。
修复方法：lambda x, i=i: i * x。
"""

# 错误版本
def make_multipliers_buggy():
    return [lambda x: i * x for i in range(5)]

# 修复版本
def make_multipliers_fixed():
    return [lambda x, i=i: i * x for i in range(5)]


if __name__ == "__main__":
    print("buggy :", [m(2) for m in make_multipliers_buggy()])   # [8, 8, 8, 8, 8]
    print("fixed :", [m(2) for m in make_multipliers_fixed()])   # [0, 2, 4, 6, 8]
