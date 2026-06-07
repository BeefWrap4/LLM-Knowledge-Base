# ---
# chapter: Ch06
# topic: sys.getrefcount 演示引用计数变化
# section: 6.1.3 引用计数的核心机制
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 03_getrefcount_demo.py
# expected_runtime: < 1s
# expected_output: 打印对象在不同操作下的引用计数 (需手动减 1)
# ---
# See: ../tutorial/06_Python内存管理与垃圾回收.md#613-引用计数的核心机制
# Interview hooks:
#   1. 为什么 sys.getrefcount(obj) 的返回值要减 1?
#   2. 哪些操作会让引用计数 +1? 哪些操作会让它 -1?
#   3. 引用计数为 0 时对象一定会被立即销毁吗?
import sys


def main() -> None:
    """通过 sys.getrefcount 跟踪一个列表对象的引用计数变化。"""
    a = [1, 2, 3]
    # sys.getrefcount 自身会持有一次引用, 所以要 -1 才能得到真实计数.
    print(f"初始引用计数: {sys.getrefcount(a) - 1}")  # 1

    b = a  # 引用计数 +1
    print(f"赋值后:       {sys.getrefcount(a) - 1}")  # 2

    c = [a, a]  # 容器两次引用同一对象, 引用计数 +2
    print(f"加入列表后:   {sys.getrefcount(a) - 1}")  # 4

    del b  # 引用计数 -1
    print(f"del b 后:     {sys.getrefcount(a) - 1}")  # 3

    del c  # 容器销毁, 内部两次引用 -1
    print(f"del c 后:     {sys.getrefcount(a) - 1}")  # 1


if __name__ == "__main__":
    main()
