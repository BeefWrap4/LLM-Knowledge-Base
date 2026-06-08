# ---
# chapter: Ch06
# topic: 内存优化最佳实践 (__slots__ / 生成器 / weakref / del / lru_cache)
# section: 6.3.3 内存优化最佳实践
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 09_memory_optimization.py
# expected_runtime: < 1s
# expected_output: __slots__ 节省内存, 生成器惰性, WeakValueDictionary 自动失效, lru_cache 命中统计
# ---
# See: ../tutorial/06_Python内存管理与垃圾回收.md#633-内存优化最佳实践
# Interview hooks:
#   1. __slots__ 节省内存的原理? 有什么副作用?
#   2. 为什么生成器 (yield) 比列表推导式更省内存?
#   3. WeakValueDictionary 与普通 dict 做缓存有什么区别?
import gc
import sys
import weakref
from functools import lru_cache


# ========== 1. __slots__ 减少内存占用 ==========
class RegularClass:
    """普通类: 每个实例都有 __dict__ (哈希表)。"""

    def __init__(self, a, b, c) -> None:
        self.a = a
        self.b = b
        self.c = c


class SlotsClass:
    """使用 __slots__: 固定属性, 不再为每个实例创建 __dict__。"""

    __slots__ = ["a", "b", "c"]

    def __init__(self, a, b, c) -> None:
        self.a = a
        self.b = b
        self.c = c


def demo_slots() -> None:
    r = RegularClass(1, 2, 3)
    s = SlotsClass(1, 2, 3)
    print(f"RegularClass 大小: {sys.getsizeof(r)} bytes")
    print(f"SlotsClass 大小:   {sys.getsizeof(s)} bytes")
    # SlotsClass 节省约 50%+ 内存


# ========== 2. 生成器 vs 列表 ==========
def get_all_data_bad(n: int):
    return [i**2 for i in range(n)]  # 一次性展开


def get_all_data_good(n: int):
    for i in range(n):
        yield i**2  # 惰性


def demo_generator() -> None:
    gen = get_all_data_good(5)
    print("生成器前 3 个值:", list(gen)[:3] if False else [next(gen) for _ in range(3)])


# ========== 3. WeakValueDictionary ==========
def demo_weakref_cache() -> None:
    cache: "weakref.WeakValueDictionary[str, Data]" = weakref.WeakValueDictionary()

    class Data:
        pass

    data = Data()
    cache["key"] = data

    print('"key" in cache (前):', "key" in cache)  # True
    del data
    gc.collect()
    print('"key" in cache (后):', "key" in cache)  # False


# ========== 4. 主动 del 大对象 ==========
def process_large_data() -> int:
    large_data = list(range(10_000_000))
    result = sum(large_data)
    del large_data  # 主动释放
    gc.collect()  # 提示 GC (可选)
    return result


# ========== 5. lru_cache ==========
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def main() -> None:
    print("--- __slots__ ---")
    demo_slots()

    print("--- 生成器 ---")
    demo_generator()

    print("--- WeakValueDictionary ---")
    demo_weakref_cache()

    print("--- process_large_data ---")
    print("sum(0..9_999_999) =", process_large_data())

    print("--- lru_cache ---")
    print("fib(100) =", fibonacci(100))
    print("缓存信息:", fibonacci.cache_info())
    fibonacci.cache_clear()


if __name__ == "__main__":
    main()
