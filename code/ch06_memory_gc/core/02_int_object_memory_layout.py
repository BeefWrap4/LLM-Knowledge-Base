# ---
# chapter: Ch06
# topic: 整数对象的内存布局与小整数缓存
# section: 6.1.2 对象的内存布局（PyObject 头部）
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 02_int_object_memory_layout.py
# expected_runtime: < 1s
# expected_output: 展示小整数缓存与列表内共享对象引用
# ---
# See: ../tutorial/06_Python内存管理与垃圾回收.md#612-对象的内存布局pyobject-头部
# Interview hooks:
#   1. Python 的 PyObject 头部包含哪些字段? 各自占用多少字节?
#   2. 为什么 a=42, b=42 时 a is b 为 True, 而 257 不是?
#   3. 列表 lst = [a, b] 存储的是值还是引用? 修改 lst[0] 会影响 b 吗?
def main() -> None:
    """演示 PyObject 行为: 整数缓存、列表存储引用。"""
    # PyObject_HEAD 由 ob_refcnt (Py_ssize_t) + ob_type 指针组成,
    # 64 位平台上各占 8 字节, 共 16 字节头部.
    # 小整数 (默认 -5 ~ 256) 在解释器启动时被缓存, 不重复创建.
    a = 42
    b = 42
    c = 257
    d = 257

    print("a is b (42):    ", a is b)   # True  命中小整数缓存
    print("c is d (257):   ", c is d)   # False 大整数不缓存 (CPython 实现细节)

    # 列表底层是 PyObject* 指针数组, 不是值拷贝.
    lst = [1, 2, 3]
    print("list:           ", lst)

    inner = [10, 20]
    lst2 = [inner, inner]   # 两个槽位都指向同一 inner 对象
    print("lst2 before:    ", lst2)

    lst2[0][0] = 99         # 通过第一个引用修改 inner
    print("lst2 after:     ", lst2)  # [[99, 20], [99, 20]] 体现共享


if __name__ == "__main__":
    main()
    print("OK")
