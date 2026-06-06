# ---
# chapter: Ch06
# topic: 内存泄漏检测 (tracemalloc / objgraph 思路 / 上下文管理器)
# section: 6.3.2 内存泄漏检测工具
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 08_memory_leak_detection.py
# expected_runtime: < 2s
# expected_output: tracemalloc 打印 TOP 分配、各类对象增长量、contextmanager 释放
# ---
# See: ../tutorial/06_Python内存管理与垃圾回收.md#632-内存泄漏检测工具
# Interview hooks:
#   1. tracemalloc 的 snapshot 对比如何定位泄漏点?
#   2. 如何在生产环境最小开销地做内存监控?
#   3. 上下文管理器如何保证资源一定被释放?
import gc
import tracemalloc
from contextlib import contextmanager


def trace_memory() -> None:
    """用 tracemalloc 追踪内存分配, 找到 TOP 分配位置。"""
    tracemalloc.start()

    snapshot1 = tracemalloc.take_snapshot()
    leak_list: list[str] = []
    for _ in range(10_000):
        leak_list.append("x" * 1000)
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    print("[内存分配 TOP 10]")
    for stat in top_stats[:10]:
        print(f"  {stat}")

    current, peak = tracemalloc.get_traced_memory()
    print(f"\n当前内存: {current / 1024 / 1024:.2f} MB")
    print(f"峰值内存: {peak / 1024 / 1024:.2f} MB")

    tracemalloc.stop()


def find_growth() -> None:
    """模拟 objgraph 的 growth 思路: 比较两次 gc.get_objects() 之间的类型增长。"""
    gc.collect()

    def snapshot() -> dict[str, int]:
        counts: dict[str, int] = {}
        for obj in gc.get_objects():
            t = type(obj).__name__
            counts[t] = counts.get(t, 0) + 1
        return counts

    before = snapshot()
    # 模拟业务操作: 大量创建 list 与 dict
    container = [[i, str(i)] for i in range(5_000)]
    del container
    gc.collect()
    after = snapshot()

    growth = {
        t: after.get(t, 0) - before.get(t, 0)
        for t in set(before) | set(after)
    }
    print("[对象增长 TOP 10]")
    for obj_type, count in sorted(growth.items(), key=lambda x: -x[1])[:10]:
        if count > 0:
            print(f"  {obj_type}: +{count}")


def find_cycle_refs() -> None:
    """使用 DEBUG_SAVEALL 把不可达对象保留到 gc.garbage 中观察。"""
    gc.set_debug(gc.DEBUG_SAVEALL)
    gc.garbage.clear()
    gc.collect()

    if gc.garbage:
        print(f"发现 {len(gc.garbage)} 个循环引用对象:")
        for obj in gc.garbage:
            print(f"  {type(obj).__name__}: {repr(obj)[:100]}")
    else:
        print("未发现循环引用")

    gc.set_debug(0)


@contextmanager
def managed_resource(name: str):
    """演示上下文管理器: 即便使用方抛异常, 资源也会在 finally 释放。"""
    resource = {"name": name, "data": []}
    try:
        yield resource
    finally:
        resource["data"].clear()
        print(f"资源 {name} 已清理")


def safe_operation() -> list[int]:
    with managed_resource("db_connection") as res:
        res["data"].extend([1, 2, 3])
        return res["data"]


def main() -> None:
    print("--- trace_memory ---")
    trace_memory()
    print("--- find_growth ---")
    find_growth()
    print("--- find_cycle_refs ---")
    find_cycle_refs()
    print("--- safe_operation ---")
    print("result:", safe_operation())


if __name__ == "__main__":
    main()
    print("OK")
