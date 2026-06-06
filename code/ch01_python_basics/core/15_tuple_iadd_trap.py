# ---
# chapter: 1
# topic: 元组 += 面试陷阱
# section: 1.3.4
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 15_tuple_iadd_trap.py
# expected_runtime: <1s
# expected_output: 演示 __iadd__ 内部机制
# ---
# See: ../tutorial/01_Python编程基础.md (lines 866-880)
# Interview hooks:
#   1. a[2] += [5, 6] 在 a=(1,2,[3,4]) 上执行会发生什么?
#   2. __iadd__ 与普通赋值的区别?
#   3. 为什么 += 会先成功再报错?
"""
🎯 面试真题：请解释以下代码的输出

a = (1, 2, [3, 4])
a[2] += [5, 6]   # 会报错吗？a 的值会变吗？

答案解析：这行代码会抛出 TypeError: 'tuple' object does not support item assignment。
但是！由于 += 操作会先执行 __iadd__（原地修改列表成功），
然后再尝试赋值给元组（失败），所以列表实际上已经被修改了。
"""

# 演示：会报 TypeError，但列表已被修改
a = (1, 2, [3, 4])
try:
    a[2] += [5, 6]
except TypeError as e:
    print(f"捕获到 TypeError: {e}")
print(f"最终结果: {a}")  # (1, 2, [3, 4, 5, 6]) — 列表已被修改！

if __name__ == "__main__":
    print("OK")
