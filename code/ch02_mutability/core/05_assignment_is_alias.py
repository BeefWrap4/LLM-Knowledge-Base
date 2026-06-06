# ---
# chapter: 2
# topic: 赋值 = 引用绑定(不是复制)
# section: 2.1.5
# difficulty: ⭐⭐
# tier: core
# deps: 无
# run: python 05_assignment_is_alias.py
# expected_runtime: <1s
# expected_output: 演示 b = a 不是复制,而是别名
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#2-1-5-内存中的对象引用关系图解
# Interview hooks:
#   1. Python 中 "赋值" 的本质是什么?变量是"盒子"还是"标签"?
#   2. b = a 之后修改 a(或 b)的内容,对方会跟着变吗?为什么?
#   3. 何时需要显式 copy()?

"""
Python 内存模型 —— 对象引用关系

关键概念:变量不是盒子,是标签!
"""

# ─────────────────────────────────────────────────────────────
# 赋值 = 引用绑定(不是复制!)
# ─────────────────────────────────────────────────────────────

a = [1, 2, 3]    # 创建列表对象,a 绑定到它
b = a            # b 绑定到同一个对象!不是复制!
b.append(4)
print(a)         # [1, 2, 3, 4] — a 也被修改了!
print(a is b)    # True

if __name__ == "__main__":
    print("OK")
