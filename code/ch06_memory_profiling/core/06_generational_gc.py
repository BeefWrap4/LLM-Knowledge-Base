# ---
# chapter: 6
# topic: Python 内存管理与性能诊断
# topic_id: memory_profiling.generational_gc
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 06_generational_gc.py
# expected_runtime: < 1s
# expected_output: 打印默认阈值、按代回收触发计数
# ---
# See: ../../../06_Python内存管理与性能诊断.md
# Interview hooks:
#   1. 分代回收背后的弱代假说是什么?
#   2. Python 3.14 的两代 GC 与 3.13 及以前有何区别?
#   3. 为什么不能把 gc.get_threshold() 的值当成跨版本固定常量?
import gc
import sys


def main() -> None:
    """展示分代回收的阈值、计数与按代触发。"""
    print(f"Python 版本:    {sys.version_info.major}.{sys.version_info.minor}")
    print("GC 阈值:        ", gc.get_threshold())
    print("GC 计数:        ", gc.get_count())

    if sys.version_info >= (3, 14):
        print("运行时语义: Python 3.14+ 只有年轻代(0)与老年代(2)；第 1 代已移除。")
        print("collect(1) 会执行年轻代收集并递增扫描一部分老年代；threshold2 已被忽略。")
    else:
        print("运行时语义: Python 3.13 及以前使用 0/1/2 三代；阈值控制各代收集频率。")

    # collect() 的 generation 参数在版本间保留，但 generation=1 的语义已变化。
    collected_0 = gc.collect(0)
    collected_1 = gc.collect(1)
    collected_2 = gc.collect(2)
    print(f"collect(0)={collected_0}, collect(1)={collected_1}, collect(2)={collected_2}")
    print("OK")


if __name__ == "__main__":
    main()
