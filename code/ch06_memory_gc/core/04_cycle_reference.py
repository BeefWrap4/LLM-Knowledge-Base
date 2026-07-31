# ---
# chapter: Ch06
# topic: 循环引用导致引用计数无法归零
# section: 6.2.2 循环引用问题与标记-清除
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 04_cycle_reference.py
# expected_runtime: < 1s
# expected_output: 演示引用计数不能独立处理循环，但循环 GC 可安全终结并回收
# ---
# See: ../tutorial/06_Python内存管理与垃圾回收.md#622-循环引用问题与标记-清除
# Interview hooks:
#   1. 为什么循环引用会让引用计数无法归零?
#   2. 引用计数机制和标记-清除算法分别负责什么?
#   3. 存在循环引用且对象定义了 __del__ 时, gc 会怎么处理?
import gc
import weakref


class Node:
    """双向链表节点, 演示 a.next ↔ b.next 形成循环引用。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.next = None

    def __del__(self) -> None:
        # PEP 442 之后，Python 终结器通常也能在循环 GC 中安全执行。
        print(f"Node {self.name} 被销毁")


def main() -> None:
    a = Node("A")
    b = Node("B")
    a_ref = weakref.ref(a)
    b_ref = weakref.ref(b)
    a.next = b  # A -> B
    b.next = a  # B -> A   形成循环引用

    # 删除外部引用后，引用计数本身无法归零；显式收集使示例可重复。
    del a
    del b
    collected = gc.collect()
    assert a_ref() is None and b_ref() is None
    print(f"循环 GC 回收完成，gc.collect() 返回 {collected}")


if __name__ == "__main__":
    main()
    print("OK")
