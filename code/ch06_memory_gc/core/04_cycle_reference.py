# ---
# chapter: Ch06
# topic: 循环引用导致引用计数无法归零
# section: 6.2.2 循环引用问题与标记-清除
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 04_cycle_reference.py
# expected_runtime: < 1s
# expected_output: 演示循环引用使 __del__ 不会被自动调用
# ---
# See: ../tutorial/06_Python内存管理与垃圾回收.md#622-循环引用问题与标记-清除
# Interview hooks:
#   1. 为什么循环引用会让引用计数无法归零?
#   2. 引用计数机制和标记-清除算法分别负责什么?
#   3. 存在循环引用且对象定义了 __del__ 时, gc 会怎么处理?
class Node:
    """双向链表节点, 演示 a.next ↔ b.next 形成循环引用。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.next = None

    def __del__(self) -> None:
        # 引用计数无法触发 __del__, 需要后续 gc.collect() 才能回收.
        print(f"Node {self.name} 被销毁")


def main() -> None:
    a = Node("A")
    b = Node("B")
    a.next = b  # A -> B
    b.next = a  # B -> A   形成循环引用

    # 删除外部引用后, 两个对象仍然互相引用, refcnt 各为 1
    del a  # 不打印 "Node A 被销毁"
    del b  # 不打印 "Node B 被销毁"
    print("外部引用已删除, 循环引用对象仍存在 (引用计数 = 1)")
    print("需要 mark-and-sweep (gc.collect) 才能回收")


if __name__ == "__main__":
    main()
