# ---
# chapter: Ch06
# topic: gc 模块的手动控制与 weakref 弱引用
# section: 6.2.6 gc 模块的手动控制
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 07_gc_module_control.py
# expected_runtime: < 1s
# expected_output: 演示 gc.disable/enable/collect、get_referrers/referents、weakref 父子关系
# ---
# See: ../tutorial/06_Python内存管理与垃圾回收.md#626-gc-模块的手动控制
# Interview hooks:
#   1. 什么场景下需要 gc.disable()? 什么时候需要重新 gc.enable()?
#   2. gc.get_referrers 与 gc.get_referents 区别?
#   3. weakref 如何打破父子节点之间的循环引用?
import gc
import weakref


def demo_gc_control() -> None:
    """展示 gc 模块的开关、调试标志、对象查询 API。"""
    # 1. 禁用 / 启用自动 GC
    gc.disable()
    print(f"GC 是否启用: {gc.isenabled()}")  # False
    gc.collect()              # 手动触发全量 GC
    gc.enable()
    print(f"GC 是否启用: {gc.isenabled()}")  # True

    # 2. 调试标志
    gc.set_debug(gc.DEBUG_STATS)        # 打印 GC 统计
    # gc.set_debug(gc.DEBUG_LEAK)       # 打印循环引用泄漏
    # gc.set_debug(gc.DEBUG_SAVEALL)    # 把不可达对象存到 gc.garbage
    # DEBUG_STATS 会向 stderr 持续输出, 演示结束后恢复默认设置
    gc.set_debug(0)

    # 3. 对象关系查询
    class MyClass:
        pass

    obj = MyClass()
    referrers = gc.get_referrers(obj)
    print(f"引用 obj 的对象数: {len(referrers)}")

    referents = gc.get_referents(obj)
    print(f"obj 引用的对象: {referents}")

    # CPython 没有 gc.get_generation(obj) 这一 API; 教程原文笔误.
    # 实际可用 gc.get_objects(generation) 列出某代对象, 这里改为兼容演示.
    obj_id = id(obj)
    for gen in (0, 1, 2):
        in_gen = any(id(o) == obj_id for o in gc.get_objects(gen))
        if in_gen:
            print(f"obj 在第 {gen} 代中")
            break
    else:
        print("obj 所属代: 已被回收或位于永久代")


def demo_weakref_parent_child() -> None:
    """父子节点用 weakref 打破循环引用的标准模式。"""
    class Parent:
        def __init__(self, name: str) -> None:
            self.name = name
            self.children: list = []

        def add_child(self, child: "Child") -> None:
            self.children.append(child)         # Parent -> Child (强引用)
            child.parent = weakref.ref(self)    # Child  -> Parent (弱引用)

    class Child:
        def __init__(self, name: str) -> None:
            self.name = name
            self.parent = None

        def get_parent(self) -> str:
            # 弱引用需要 () 解引用
            parent = self.parent() if self.parent else None
            return parent.name if parent else "已销毁"

    parent = Parent("P1")
    child = Child("C1")
    parent.add_child(child)

    print(f"child 的 parent: {child.get_parent()}")  # P1
    del parent
    print(f"parent 销毁后:    {child.get_parent()}")  # 已销毁


def main() -> None:
    demo_gc_control()
    demo_weakref_parent_child()


if __name__ == "__main__":
    main()
    print("OK")
