# ---
# chapter: 8
# topic: Python 数据结构与算法
# topic_id: data_structures.linked_list
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 01_linked_list.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../../../08_Python数据结构与算法.md
#
# Interview hooks:
#  1. 反转链表（LeetCode 206）：如何用迭代法和递归法反转单链表？时间/空间复杂度？
#  2. 环形链表 II（LeetCode 142）：如何用快慢指针找到环的入口节点？数学证明是 a = c + kb？
#  3. LRU Cache（LeetCode 146）：为什么必须用双向链表 + 哈希表？单链表能否实现 O(1) 删除？


class ListNode:
    """单向链表节点"""

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class LinkedList:
    """链表操作工具类"""

    @staticmethod
    def create(arr: list) -> ListNode:
        """从数组创建链表"""
        dummy = ListNode(0)
        curr = dummy
        for val in arr:
            curr.next = ListNode(val)
            curr = curr.next
        return dummy.next

    @staticmethod
    def to_list(head: ListNode) -> list:
        """链表转数组"""
        result = []
        while head:
            result.append(head.val)
            head = head.next
        return result


# ========== 面试高频题1：反转链表 ==========
def reverse_list(head: ListNode) -> ListNode:
    """
    迭代法反转链表
    时间复杂度: O(n)  空间复杂度: O(1)
    """
    prev, curr = None, head
    while curr:
        next_temp = curr.next  # 暂存下一个节点
        curr.next = prev  # 反转当前指针
        prev = curr  # prev 前移
        curr = next_temp  # curr 前移
    return prev  # 新的头节点


# ========== 面试高频题2：判断环形链表 ==========
def has_cycle(head: ListNode) -> bool:
    """
    快慢指针判断环形链表
    时间复杂度: O(n)  空间复杂度: O(1)
    """
    if not head or not head.next:
        return False

    slow = head  # 慢指针：每次走1步
    fast = head.next  # 快指针：每次走2步

    while fast and fast.next:
        if slow == fast:  # 相遇则有环
            return True
        slow = slow.next
        fast = fast.next.next

    return False  # 快指针到达末尾，无环


def detect_cycle_entry(head: ListNode) -> ListNode:
    """
    找到环的入口节点
    时间复杂度: O(n)  空间复杂度: O(1)

    原理：相遇后，一个指针回到头，两个指针同步走，再次相遇即入口
    """
    if not head or not head.next:
        return None

    # 阶段1：判断是否有环，找到相遇点
    slow = fast = head
    has_cycle = False
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            has_cycle = True
            break

    if not has_cycle:
        return None

    # 阶段2：找入口
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow  # 环的入口


# ========== 面试高频题3：合并两个有序链表 ==========
def merge_two_lists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    合并两个有序链表
    时间复杂度: O(m+n)  空间复杂度: O(1)
    """
    dummy = ListNode(0)
    curr = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    # 连接剩余部分
    curr.next = l1 if l1 else l2
    return dummy.next


# ========== 面试高频题4：LRU 缓存 ==========
class DLinkedNode:
    """双向链表节点，用于 LRU Cache"""

    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache:
    """
    LRU 缓存实现（哈希表 + 双向链表）

    时间复杂度：
      - get: O(1)
      - put: O(1)
    空间复杂度：O(capacity)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> DLinkedNode
        self.size = 0

        # 伪头部和伪尾部节点，简化边界处理
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_head(node)  # 标记为最近使用
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._move_to_head(node)
        else:
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            self.size += 1

            if self.size > self.capacity:
                removed = self._remove_tail()
                del self.cache[removed.key]
                self.size -= 1

    # --- 双向链表辅助方法 ---
    def _add_to_head(self, node: DLinkedNode):
        """在头部添加节点"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DLinkedNode):
        """移除节点"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: DLinkedNode):
        """将节点移到头部"""
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_tail(self) -> DLinkedNode:
        """移除尾部节点（最久未使用）"""
        node = self.tail.prev
        self._remove_node(node)
        return node


if __name__ == "__main__":
    # 测试反转链表
    head = LinkedList.create([1, 2, 3, 4, 5])
    reversed_head = reverse_list(head)
    assert LinkedList.to_list(reversed_head) == [5, 4, 3, 2, 1]

    # 测试合并有序链表
    l1 = LinkedList.create([1, 2, 4])
    l2 = LinkedList.create([1, 3, 4])
    merged = merge_two_lists(l1, l2)
    assert LinkedList.to_list(merged) == [1, 1, 2, 3, 4, 4]

    # 测试 LRU Cache
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)  # 淘汰 key=2
    assert cache.get(2) == -1
    cache.put(4, 4)  # 淘汰 key=1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4


print("OK")
