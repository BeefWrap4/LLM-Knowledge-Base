# ---
# chapter: 2
# topic: Python 对象模型与可变性
# topic_id: object_model.custom_deepcopy
# difficulty: ⭐⭐
# tier: core
# deps: copy
# run: python 12_custom_deepcopy.py
# expected_runtime: <1s
# expected_output: 演示自定义 __deepcopy__,深拷贝 value 但共享 cache
# ---
# See: ../../../02_Python对象模型与可变性.md
# Interview hooks:
#   1. 哪些对象无法被 deepcopy?为什么?
#   2. 自定义类如何控制 deepcopy 行为?__deepcopy__ 的 memo 参数是什么?
#   3. 什么场景下需要自定义 __deepcopy__?(如共享缓存/资源句柄)

"""
deepcopy 的限制与自定义 —— 面试加分项
"""

import copy
import tempfile

# ─────────────────────────────────────────────────────────────
# 深拷贝的限制
# ─────────────────────────────────────────────────────────────

def demonstrate_noncopyable_resource() -> None:
    """用自动清理的临时文件演示资源句柄不能 deepcopy。"""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as file_handle:
        try:
            copy.deepcopy(file_handle)
        except TypeError as exc:
            print(f"文件句柄 deepcopy 失败（符合预期）: {type(exc).__name__}")

# 2. 不能拷贝函数、模块、类型本身
# copy.deepcopy(lambda x: x)   # 通常可以但结果可能是同一个对象

# 3. 自定义对象默认只拷贝 __dict__

# ─────────────────────────────────────────────────────────────
# 自定义深拷贝行为:__deepcopy__
# ─────────────────────────────────────────────────────────────


class Node:
    """链表节点 — 自定义深拷贝行为"""

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node
        # 共享数据(不需要深拷贝)
        self.shared_cache = {"metadata": "共享元数据"}

    def __deepcopy__(self, memo):
        """自定义深拷贝逻辑"""
        # 创建新节点,但只拷贝 value,共享 cache
        new_node = Node(self.value)
        memo[id(self)] = new_node

        if self.next:
            new_node.next = copy.deepcopy(self.next, memo)

        # 共享 cache(不创建新的)
        new_node.shared_cache = self.shared_cache

        return new_node

    def __repr__(self):
        return f"Node({self.value})"


# 构建链表 1 -> 2 -> 3
node3 = Node(3)
node2 = Node(2, node3)
node1 = Node(1, node2)

node1_copy = copy.deepcopy(node1)

print(f"值独立? {node1_copy.value == node1.value and node1_copy is not node1}")  # True
print(f"next独立? {node1_copy.next is not node1.next}")  # True
print(f"cache共享? {node1_copy.shared_cache is node1.shared_cache}")  # True — 故意共享

if __name__ == "__main__":
    demonstrate_noncopyable_resource()
    print("OK")
