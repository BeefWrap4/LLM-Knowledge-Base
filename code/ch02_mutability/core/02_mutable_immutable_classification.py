# ---
# chapter: 2
# topic: 可变与不可变类型分类与修改行为
# section: 2.1.2
# difficulty: ⭐⭐
# tier: core
# deps: 无
# run: python 02_mutable_immutable_classification.py
# expected_runtime: <1s
# expected_output: 演示可变与不可变类型的修改行为差异(id 变化)
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#2-1-2-完整类型分类表
# Interview hooks:
#   1. Python 中哪些类型是可变的,哪些是不可变的?
#   2. 不可变对象"修改"后 id 为什么会变?可变对象呢?
#   3. 为什么不可变类型可以作为 dict 键,可变类型不行?

"""
Python 数据类型的可变与不可变完整分类
"""

# ─────────────────────────────────────────────────────────────
# 不可变类型(Immutable)
# ─────────────────────────────────────────────────────────────

# 1. 数值类型
n = 42  # int
f = 3.14  # float
c = 3 + 4j  # complex

# 2. 字符串
s = "hello"  # str

# 3. 元组
t = (1, 2, 3)  # tuple

# 4. 冻结集合
fs = frozenset([1, 2, 3])

# 5. 字节串
b = b"bytes"  # bytes

# ─────────────────────────────────────────────────────────────
# 可变类型(Mutable)
# ─────────────────────────────────────────────────────────────

# 1. 列表
lst = [1, 2, 3]  # list

# 2. 字典
d = {"a": 1}  # dict

# 3. 集合
se = {1, 2, 3}  # set

# 4. 字节数组
ba = bytearray(b"hello")

# ─────────────────────────────────────────────────────────────
# 修改行为对比(面试核心考点)
# ─────────────────────────────────────────────────────────────


def demo_mutable_vs_immutable():
    """演示可变与不可变的修改行为差异"""

    # ── 不可变对象:修改 = 创建新对象 ──
    x = 10
    old_id = id(x)
    x += 1  # x 现在绑定到新对象 11
    new_id = id(x)
    print(f"int 修改: id 从 {old_id} 变为 {new_id} — {'不同对象!' if old_id != new_id else '同一对象'}")

    s = "hello"
    old_id = id(s)
    s += " world"  # 创建新字符串
    new_id = id(s)
    print(f"str 修改: id 从 {old_id} 变为 {new_id} — {'不同对象!' if old_id != new_id else '同一对象'}")

    # ── 可变对象:修改 = 原地修改 ──
    lst = [1, 2, 3]
    old_id = id(lst)
    lst.append(4)  # 原地修改,id 不变
    new_id = id(lst)
    print(f"list 修改: id 从 {old_id} 变为 {new_id} — {'同一对象' if old_id == new_id else '不同对象!'}")
    print(f"  修改后: {lst}")


if __name__ == "__main__":
    demo_mutable_vs_immutable()
