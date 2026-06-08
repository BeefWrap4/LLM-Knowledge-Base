# ---
# chapter: 4
# topic: Python高级特性与函数式编程
# section: 4.3.1 迭代器协议
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 09_iterator_protocol.py
# expected_runtime: <1s
# expected_output: 倒计时序列 + 多次遍历 vs 单次遍历对比
# ---
# See: ../tutorial/04_Python高级特性与函数式编程.md — §4.3.1 迭代器协议
# Interview hooks:
#   1. 可迭代对象（Iterable）和迭代器（Iterator）的区别？
#   2. for 循环背后的迭代器协议是如何工作的？
#   3. 为什么 StopIteration 异常被设计为迭代结束的信号？

"""
迭代器（Iterator）协议 —— Python 的遍历基础

迭代器必须实现：
  __iter__()    → 返回迭代器自身
  __next__()    → 返回下一个元素，没有时抛出 StopIteration

可迭代对象（Iterable）：实现 __iter__()，返回一个迭代器
"""

# ─────────────────────────────────────────────────────────────
# 手写迭代器 —— 理解底层机制
# ─────────────────────────────────────────────────────────────


class CountDown:
    """倒计时迭代器 —— 从 n 数到 1"""

    def __init__(self, start):
        self.start = start

    def __iter__(self):
        """返回迭代器对象（自身）"""
        self.current = self.start  # 重置计数器
        return self

    def __next__(self):
        """返回下一个值"""
        if self.current <= 0:
            raise StopIteration  # 迭代结束的信号
        num = self.current
        self.current -= 1
        return num


# 使用
countdown = CountDown(5)
for n in countdown:
    print(n, end=" ")  # 5 4 3 2 1
print()

# 等价于：
countdown = CountDown(3)
iterator = iter(countdown)  # 调用 __iter__()
for _ in range(5):  # 用 for 循环自动处理 StopIteration (PEP 479)
    try:
        print(next(iterator))  # 5, 4, 3, 2, 1
    except StopIteration:
        print("(iteration done)")
        break
# next(iterator)             # StopIteration

# ─────────────────────────────────────────────────────────────
# 迭代器 vs 可迭代对象
# ─────────────────────────────────────────────────────────────

# 可迭代对象可多次遍历
lst = [1, 2, 3]
for x in lst:
    print(x, end=" ")  # 1 2 3
for x in lst:
    print(x, end=" ")  # 1 2 3 — 再次遍历

# 迭代器只能遍历一次
it = iter([1, 2, 3])
for x in it:
    print(x, end=" ")  # 1 2 3
for x in it:
    print(x, end=" ")  # 无输出 — 迭代器已耗尽！


if __name__ == "__main__":
    print("OK")
