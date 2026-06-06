# ---
# chapter: 1
# topic: 字典 Dict — 哈希表实现
# section: 1.3.2
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 12_dict_operations.py
# expected_runtime: <1s
# expected_output: 字典操作示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 589-730)
# Interview hooks:
#   1. 字典查找为什么平均是 O(1)?最坏情况呢?
#   2. Python 3.6+ 紧凑字典的 entries/indices 双数组结构?
#   3. 自定义类作为字典键需要实现哪些方法?
"""
字典：Python 最核心的数据结构
底层实现：开放寻址法 + 伪删除标记（Python 3.6+ 使用紧凑字典，保持插入顺序）

查找时间复杂度：平均 O(1)，最坏 O(n)（哈希冲突严重时）
"""

# ─────────────────────────────────────────────────────────────
# 字典创建与常用操作
# ─────────────────────────────────────────────────────────────

# 多种创建方式
d1 = {"a": 1, "b": 2}                          # 字面量
d2 = dict(a=1, b=2)                             # 关键字参数
d3 = dict([("a", 1), ("b", 2)])                 # 键值对序列
d4 = {k: v for k, v in [("a", 1), ("b", 2)]}    # 字典推导式

# ─────────────────────────────────────────────────────────────
# get / setdefault / defaultdict（面试常考对比）
# ─────────────────────────────────────────────────────────────

def count_words(words: list) -> dict:
    """
    三种 word count 写法对比
    """
    # 写法1：传统方式
    count1 = {}
    for word in words:
        if word in count1:
            count1[word] += 1
        else:
            count1[word] = 1

    # 写法2：get 方法
    count2 = {}
    for word in words:
        count2[word] = count2.get(word, 0) + 1

    # 写法3：setdefault（不常用，面试可能问）
    count3 = {}
    for word in words:
        count3.setdefault(word, 0)
        count3[word] += 1

    # 写法4：collections.Counter（最 Pythonic）
    from collections import Counter
    count4 = Counter(words)

    # 写法5：defaultdict（面试推荐写法）
    from collections import defaultdict
    count5 = defaultdict(int)
    for word in words:
        count5[word] += 1

    return dict(count5)

print(f"word count: {count_words(['a', 'b', 'a', 'c', 'a', 'b'])}")

# ─────────────────────────────────────────────────────────────
# 字典合并（Python 3.9+ 语法）
# ─────────────────────────────────────────────────────────────

def merge_dicts(d1: dict, d2: dict) -> dict:
    """字典合并的多种方式"""

    # Python 3.9+：合并运算符
    merged = d1 | d2        # 创建新字典，d2 的键覆盖 d1
    d1 |= d2                # 原地更新 d1

    # Python 3.5+：解包合并
    merged = {**d1, **d2}

    # 传统方式
    merged = d1.copy()
    merged.update(d2)

    return merged

# ─────────────────────────────────────────────────────────────
# 字典哈希表原理（面试核心考点）
# ─────────────────────────────────────────────────────────────

"""
字典查找 O(1) 的原理：

┌─────────────────────────────────────────────┐
│              哈希表查找流程                    │
│                                             │
│  键 key                                     │
│   │                                         │
│   ▼                                         │
│  hash(key) ──→ 哈希值                       │
│   │                                         │
│   ▼                                         │
│  哈希值 % 表大小 ──→ 索引位置                 │
│   │                                         │
│   ▼                                         │
│  检查该位置：                                │
│    - 为空 → KeyError                        │
│    - 键匹配 → 返回值                         │
│    - 键不匹配（冲突）→ 探测下一个位置           │
│                                             │
└─────────────────────────────────────────────┘

Python 3.6+ 紧凑字典结构：
- entries 数组：按插入顺序存储 [hash, key, value]
- indices 数组：哈希表，存储 entries 的索引
- 这使得字典既有 O(1) 查找，又天然保持插入顺序
"""

# 字典键的要求：必须是不可变且可哈希的
# 可变类型（list, dict, set）不能作为字典键
# tuple 只有在元素全部不可变时才能作为键

valid_key = (1, "a", (2, 3))   # ✅ 嵌套 tuple 元素都是不可变的
# invalid_key = (1, [2, 3])    # ❌ 包含 list，不可哈希

# 自定义类作为键：需要实现 __hash__ 和 __eq__
class HashablePoint:
    """可作为字典键的二维点"""
    __slots__ = ["x", "y"]   # 节省内存（面试加分项）

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))   # 基于不可变元组

    def __eq__(self, other):
        if not isinstance(other, HashablePoint):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"HashablePoint({self.x}, {self.y})"

points = {
    HashablePoint(0, 0): "原点",
    HashablePoint(1, 1): "(1,1)点",
}
print(points[HashablePoint(0, 0)])  # "原点"

if __name__ == "__main__":
    print("OK")
