# ---
# chapter: 2
# topic: Python 对象模型与可变性
# topic_id: python_runtime.list_operations
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 11_list_operations.py
# expected_runtime: <1s
# expected_output: 列表操作示例
# ---
# See: ../../../02_Python对象模型与可变性.md
# Interview hooks:
#   1. 列表的 append/pop/pop(0) 时间复杂度分别是?
#   2. 列表推导式与生成器表达式的内存差异?
#   3. 列表去重方法中,哪种既保顺序又高效?
"""
列表：面试高频操作与复杂度分析
底层实现：过度分配的动态数组（PyListObject）
"""

# ─────────────────────────────────────────────────────────────
# 时间复杂度速查
# ─────────────────────────────────────────────────────────────
# 操作                  复杂度        说明
# ──────────────────────────────────────────────
# lst[i]                O(1)         随机访问
# lst.append(x)         均摊 O(1)    可能触发扩容
# lst.pop()             O(1)         尾部弹出
# lst.pop(0)            O(n)         头部弹出（元素全移动）
# lst.insert(i, x)      O(n)         中间插入
# lst[i:j] = [...]      O(n)         切片赋值
# x in lst              O(n)         线性查找
# lst.sort()            O(n log n)   Timsort 算法

# ─────────────────────────────────────────────────────────────
# 列表推导式（Pythonic 写法，面试常考）
# ─────────────────────────────────────────────────────────────

# 基础推导式
squares = [x**2 for x in range(10)]
print(f"平方: {squares}")

# 带条件过滤
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"偶数平方: {even_squares}")

# 嵌套推导式（矩阵转置）
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = [[row[i] for row in matrix] for i in range(3)]
print(f"转置: {transposed}")
# [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

# 面试陷阱：列表推导式的变量泄漏（Python 2 问题，Python 3 已修复）
# Python 3 中推导式有自己的局部作用域
x = 10
[x for x in range(5)]
print(f"x 仍然为: {x}")  # Python 3: 10（x 不变）；Python 2: 4（x 被修改）

# ─────────────────────────────────────────────────────────────
# 切片操作（slice）— 面试高频
# ─────────────────────────────────────────────────────────────

lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 切片语法：lst[start:stop:step]
print(lst[2:7])  # [2, 3, 4, 5, 6]      从索引2到6
print(lst[:4])  # [0, 1, 2, 3]          从头到3
print(lst[::2])  # [0, 2, 4, 6, 8]       步长2
print(lst[::-1])  # [9, 8, 7, ..., 0]     反转列表

# 🎯 面试陷阱：切片越界不报错
print(lst[5:100])  # [5, 6, 7, 8, 9] — 不抛异常！
# print(lst[100])     # IndexError！— 索引越界才报错


# 删除偶数索引元素（正确 vs 错误写法）
def remove_even_indices_wrong(lst):
    """❌ 错误：遍历中修改列表导致跳过元素"""
    for i, val in enumerate(lst):
        if i % 2 == 0:
            del lst[i]  # 删除后索引偏移！
    return lst


def remove_even_indices_right(lst):
    """✅ 正确：切片删除"""
    del lst[::2]  # 一次性删除所有偶数索引
    return lst


# 或者用列表推导式重建
def remove_even_indices(lst):
    return [val for i, val in enumerate(lst) if i % 2 == 1]


# ─────────────────────────────────────────────────────────────
# 列表去重的 N 种方法（按效率排序，面试常考）
# ─────────────────────────────────────────────────────────────


def deduplicate_methods(data):
    """列表去重方法对比"""
    methods = {}

    # 方法1：set 去重（最快，但不保持顺序）
    methods["set"] = list(set(data))

    # 方法2：dict.fromkeys 去重（保持顺序，Python 3.7+）
    methods["dict_fromkeys"] = list(dict.fromkeys(data))

    # 方法3：循环判断（最慢，但最直观）
    seen = set()
    result = []
    for x in data:
        if x not in seen:
            seen.add(x)
            result.append(x)
    methods["loop"] = result

    return methods


data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
results = deduplicate_methods(data)
for name, result in results.items():
    print(f"{name:15s}: {result}")

if __name__ == "__main__":
    print("OK")
