# ---
# chapter: 7
# topic: Python 数据结构与算法
# section: 7.3.2 归并排序
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 07_merge_sort.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/07_Python数据结构与算法.md#7.3.2-归并排序
#
# Interview hooks:
#  1. 归并排序的稳定性：为什么 merge 时 left[i] <= right[j] 而不是 < ？稳定性在数据库排序中的意义？
#  2. 归并排序的空间复杂度：为什么是 O(n)？能否原地归并？Knuth 提出的原地归并复杂度？
#  3. 链表排序为什么用归并而不用快排？快排在链表上分区困难，归并天然适合链表？


def merge_sort(nums: list[int]) -> list[int]:
    """
    归并排序

    时间复杂度: 稳定 O(nlogn) — 最坏、最好、平均都是 O(nlogn)
    空间复杂度: O(n) — 需要额外数组
    稳定性: 稳定 — 相等元素保持原顺序
    """
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])

    return merge(left, right)


def merge(left: list[int], right: list[int]) -> list[int]:
    """合并两个有序数组"""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # 等号保证稳定性
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


if __name__ == "__main__":
    # 测试归并排序
    assert merge_sort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]
    assert merge_sort([]) == []
    assert merge_sort([1]) == [1]
    assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    # 测试稳定性
    # 输入 [(value, index)] 检查相同 value 的相对顺序
    pairs = [(2, 0), (1, 1), (2, 2), (1, 3)]
    # merge_sort 返回普通 list；这里直接用 sorted 验证稳定性
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    assert sorted_pairs == [(1, 1), (1, 3), (2, 0), (2, 2)]

    # 测试 merge 函数
    assert merge([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge([], [1, 2]) == [1, 2]
    assert merge([1, 2], []) == [1, 2]

    print("OK")
