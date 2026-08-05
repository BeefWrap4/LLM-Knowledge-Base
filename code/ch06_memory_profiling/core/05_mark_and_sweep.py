# ---
# chapter: 6
# topic: Python 内存管理与性能诊断
# topic_id: memory_profiling.mark_and_sweep
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 05_mark_and_sweep.py
# expected_runtime: < 1s
# expected_output: 调用 gc.collect() 回收循环引用对象, 触发 __del__
# ---
# See: ../../../06_Python内存管理与性能诊断.md
# Interview hooks:
#   1. 标记-清除的两个阶段分别做什么?
#   2. 为什么 gc.collect() 不一定能回收带 __del__ 的循环引用?
#   3. gc.garbage 在什么场景下才会有内容?
import gc


class Node:
    """构造循环引用的节点。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.next = None

    def __del__(self) -> None:
        print(f"  Node {self.name} 被销毁")


def main() -> None:
    a = Node("A")
    b = Node("B")
    a.next = b
    b.next = a

    del a, b
    print("删除外部引用后, 循环引用对象仍存在")
    print(f"不可达对象数量: {len(gc.garbage)}")

    # 手动触发 mark-and-sweep, 释放循环引用
    print("手动触发 gc.collect()...")
    collected = gc.collect()  # 返回回收的对象数量
    print(f"回收了 {collected} 个对象")


if __name__ == "__main__":
    main()
    print("OK")
