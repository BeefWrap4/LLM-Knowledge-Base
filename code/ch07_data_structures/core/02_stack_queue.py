# ---
# chapter: 7
# topic: Python 数据结构与算法
# section: 7.1.2 栈与队列
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: collections
# run: python 02_stack_queue.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/07_Python数据结构与算法.md#7.1.2-栈与队列
#
# Interview hooks:
#  1. 用两个栈实现队列：push 走一个栈，pop 时另一个栈空了才把 in 栈倒入 out 栈，为什么是均摊 O(1)？
#  2. 最小栈（LeetCode 155）：如何在 O(1) 时间获取栈中最小值？辅助栈与每个元素同步压入？
#  3. 单调栈的下一个更大元素：为什么用单调递减栈？时间复杂度为什么是 O(n)？


from collections import deque

# ========== 栈（LIFO） ==========
# 使用 list 即可
stack = []
stack.append(1)    # 入栈: O(1)
stack.append(2)
top = stack[-1]    # 查看栈顶: O(1)
stack.pop()        # 出栈: O(1)

# ========== 队列（FIFO） ==========
# 使用 deque，list.pop(0) 是 O(n) 不推荐使用
queue = deque()
queue.append(1)     # 入队: O(1)
queue.append(2)
front = queue[0]    # 查看队首: O(1)
queue.popleft()     # 出队: O(1)


# ========== 面试高频题：用栈实现队列 ==========
class MyQueue:
    """
    用两个栈实现队列

    push: O(1)    pop: 均摊 O(1)    peek: 均摊 O(1)
    """
    def __init__(self):
        self.stack_in = []   # 入队栈
        self.stack_out = []  # 出队栈

    def push(self, x: int) -> None:
        self.stack_in.append(x)

    def pop(self) -> int:
        self.peek()  # 确保 out 栈有元素
        return self.stack_out.pop()

    def peek(self) -> int:
        if not self.stack_out:
            # 将 in 栈所有元素倒入 out 栈
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out[-1]

    def empty(self) -> bool:
        return not self.stack_in and not self.stack_out


# ========== 面试高频题：最小栈 ==========
class MinStack:
    """
    支持 O(1) 获取最小值的栈

    使用辅助栈存储每个状态下的最小值
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []  # 辅助栈，存储当前最小值

    def push(self, val: int) -> None:
        self.stack.append(val)
        # 辅助栈存入当前最小值
        if not self.min_stack:
            self.min_stack.append(val)
        else:
            self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


# ========== 单调栈：下一个更大元素 ==========
def next_greater_elements(nums: list[int]) -> list[int]:
    """
    找到每个元素右边第一个比它大的元素

    单调递减栈
    时间复杂度: O(n)  空间复杂度: O(n)
    """
    n = len(nums)
    result = [-1] * n  # 默认 -1 表示没有更大元素
    stack = []  # 存储索引，保持递减

    for i in range(n):
        # 当前元素比栈顶大，说明找到了"下一个更大元素"
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result


if __name__ == "__main__":
    # 测试用栈实现队列
    q = MyQueue()
    q.push(1)
    q.push(2)
    assert q.peek() == 1
    assert q.pop() == 1
    assert not q.empty()
    assert q.pop() == 2
    assert q.empty()

    # 测试最小栈
    ms = MinStack()
    ms.push(-2)
    ms.push(0)
    ms.push(-3)
    assert ms.getMin() == -3
    ms.pop()
    assert ms.top() == 0
    assert ms.getMin() == -2

    # 测试单调栈
    assert next_greater_elements([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]

