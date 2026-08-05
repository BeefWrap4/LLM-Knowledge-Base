# ---
# chapter: 2
# topic: Python 对象模型与可变性
# topic_id: object_model.immutable_reassign
# difficulty: ⭐⭐
# tier: core
# deps: 无
# run: python 06_immutable_reassign.py
# expected_runtime: <1s
# expected_output: 演示 y 不受 x 重新赋值的影响(不可变对象)
# ---
# See: ../../../02_Python对象模型与可变性.md
# Interview hooks:
#   1. x = 10; y = x; x = 20 后 y 是什么?为什么?
#   2. 不可变对象被"修改"时,实际发生了什么?
#   3. 这与"赋值即引用绑定"是否矛盾?

"""
不可变对象的重新赋值
"""

# ─────────────────────────────────────────────────────────────
# 不可变对象的重新赋值
# ─────────────────────────────────────────────────────────────

x = 10  # x 绑定到 int 对象 10
y = x  # y 绑定到同一个对象
x = 20  # x 绑定到新对象 20,y 不变
print(y)  # 10 — y 不受影响

if __name__ == "__main__":
    print("OK")
