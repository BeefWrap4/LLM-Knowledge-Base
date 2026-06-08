# ---
# chapter: 7
# topic: Python 数据结构与算法
# section: 7.1.3 哈希表
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: collections
# run: python 03_hash_table.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/07_Python数据结构与算法.md#7.1.3-哈希表
#
# Interview hooks:
#  1. 两数之和（LeetCode 1）：为什么用哈希表能在 O(n) 找到？不用哈希表用排序+双指针是 O(nlogn) 吗？
#  2. 字母异位词（LeetCode 242）：如何判断两个字符串是字母异位词？Counter 的时间/空间复杂度？
#  3. 设计哈希集合：拉链法和开放地址法处理冲突的优缺点？Python dict 用的是哪种？


# Python dict 基于哈希表实现
# 平均时间复杂度: O(1) 查找/插入/删除
# 最坏情况: O(n)（哈希冲突严重时）

# ========== 面试高频题：两数之和 ==========
def two_sum(nums: list[int], target: int) -> list[int]:
    """
    哈希表解法
    时间复杂度: O(n)  空间复杂度: O(n)
    """
    seen = {}  # 值 -> 索引
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


# ========== 面试高频题：判断字母异位词 ==========
def is_anagram(s: str, t: str) -> bool:
    """
    哈希表统计字符频率
    时间复杂度: O(n)  空间复杂度: O(1)（字符集大小固定26）
    """
    if len(s) != len(t):
        return False

    from collections import Counter

    return Counter(s) == Counter(t)


# ========== 面试高频题：设计哈希集合（拉链法） ==========
class MyHashSet:
    """
    拉链法处理哈希冲突

    时间复杂度: 平均 O(1)，最坏 O(n)（所有键冲突）
    """

    def __init__(self):
        self.base = 769  # 质数取模减少冲突
        self.data = [[] for _ in range(self.base)]

    def _hash(self, key: int) -> int:
        return key % self.base

    def add(self, key: int) -> None:
        h = self._hash(key)
        if key not in self.data[h]:
            self.data[h].append(key)

    def remove(self, key: int) -> None:
        h = self._hash(key)
        if key in self.data[h]:
            self.data[h].remove(key)

    def contains(self, key: int) -> bool:
        h = self._hash(key)
        return key in self.data[h]


if __name__ == "__main__":
    # 测试两数之和
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]

    # 测试字母异位词
    assert is_anagram("anagram", "nagaram") is True
    assert is_anagram("rat", "car") is False

    # 测试设计哈希集合
    hs = MyHashSet()
    hs.add(1)
    hs.add(2)
    assert hs.contains(1) is True
    assert hs.contains(3) is False
    hs.add(2)
    assert hs.contains(2) is True
    hs.remove(2)
    assert hs.contains(2) is False
