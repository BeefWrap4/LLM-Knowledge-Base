# ---
# chapter: Ch06
# topic: 分代回收阈值与按代触发
# section: 6.2.4 分代回收（Generational GC）
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 06_generational_gc.py
# expected_runtime: < 1s
# expected_output: 打印默认阈值、按代回收触发计数
# ---
# See: ../tutorial/06_Python内存管理与垃圾回收.md#624-分代回收generational-gc
# Interview hooks:
#   1. 分代回收背后的弱代假说是什么?
#   2. gc.get_threshold() 返回 (700, 10, 10) 各自的含义?
#   3. gc.collect(0) / gc.collect(1) / gc.collect(2) 区别?
import gc


def main() -> None:
    """展示分代回收的阈值、计数与按代触发。"""
    # (threshold0, threshold1, threshold2)
    # 700: 第 0 代分配 700 次触发一次 GC
    # 10:  第 0 代 GC 累计 10 次触发一次第 1 代 GC
    # 10:  第 1 代 GC 累计 10 次触发一次第 2 代 GC
    print("GC 阈值:        ", gc.get_threshold())
    print("各代对象数量:   ", gc.get_count())

    # 单独清理某一代
    collected_0 = gc.collect(0)  # 只清理第 0 代
    collected_1 = gc.collect(1)  # 清理第 0、1 代
    collected_2 = gc.collect(2)  # 清理所有代 (最彻底)
    print(f"collect(0)={collected_0}, collect(1)={collected_1}, collect(2)={collected_2}")


if __name__ == "__main__":
    main()
    print("OK")
