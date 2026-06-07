# ---
# chapter: 1
# topic: 列表与字典基础 (含 4 种去重方法)
# section: 1.3.1, 1.3.2
# difficulty: ⭐⭐⭐
# tier: core
# deps: stdlib
# run: python 01_list_dict_basics.py
# expected_runtime: <1s
# expected_output: 包含 "OK"
# ---
#
# See: ../tutorial/Ch01_Python编程基础.md §1.3.1 (列表) + §1.3.2 (字典)
# Interview hooks:
#   - "列表的 append 和 + 区别?"  →  O(1) vs O(n), 前者原地后者创建新对象
#   - "4 种列表去重方法?"       →  set / dict.fromkeys / loop / Counter
#   - "dict 查找为什么 O(1)?"    →  哈希表 + 开放寻址

# 列表
lst = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
print(f"原列表: {lst}")
print(f"长度: {len(lst)}, 反转: {lst[::-1]}")
print(f"列表推导式平方: {[x**2 for x in lst[:5]]}")
print(f"包含偶数: {[x for x in lst if x % 2 == 0]}")


# 4 种去重方法
def dedupe_set(data):
    """最快, 不保持顺序"""
    return list(set(data))


def dedupe_dict_fromkeys(data):
    """保持顺序, Python 3.7+ 推荐"""
    return list(dict.fromkeys(data))


def dedupe_loop(data):
    """最慢但最直观"""
    seen, result = set(), []
    for x in data:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


from collections import Counter


def dedupe_counter(data):
    """统计同时去重"""
    return list(Counter(data).keys())


methods = [dedupe_set, dedupe_dict_fromkeys, dedupe_loop, dedupe_counter]
expected_unique = {1, 2, 3, 4, 5, 6, 9}  # 7 unique values in lst
for method in methods:
    result = method(lst)
    assert set(result) == expected_unique, f"{method.__name__} 去重失败: {result}"
    print(f"  {method.__name__:25s} → {result}")


# 字典
d = {"a": 1, "b": 2, "c": 3}
print(f"\n字典: {d}")
print(f"get('d', 0): {d.get('d', 0)}")
print(f"keys: {list(d.keys())}")
print(f"反转: {dict(zip(d.values(), d.keys()))}")

# 字典合并 (Python 3.9+)
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}
merged = d1 | d2
print(f"\n合并 d1|d2: {merged}")

# 字数统计
sentence = "Python 编程基础 Python 大模型 Python 面试"
word_count = {}
for word in sentence.split():
    word_count[word] = word_count.get(word, 0) + 1
print(f"字数统计: {word_count}")

print("\nOK")

print("OK")
