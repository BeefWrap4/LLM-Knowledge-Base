# ---
# chapter: 7
# topic: Python 数据结构与算法
# section: 7.3.1 快速排序
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: random
# run: python 06_quicksort.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/07_Python数据结构与算法.md#7.3.1-快速排序
#
# Interview hooks:
#  1. 快速排序的 partition：为什么用 Lomuto 或 Hoare 分区？两种分区的时间/空间复杂度？
#  2. 快排最坏情况：为什么有序数组用第一个元素做基准是 O(n²)？随机化基准和三数取中如何避免？
#  3. 快排 vs 归并：为什么实际应用中快排更常用？稳定性与缓存友好性的取舍？


import random


def quicksort(nums: list[int]) -> list[int]:
    """
    快速排序

    时间复杂度: 平均 O(nlogn)，最坏 O(n²)
    空间复杂度: O(logn) 递归栈
    稳定性: 不稳定

    分治策略：选基准 -> 分区 -> 递归排序
    """
    if len(nums) <= 1:
        return nums

    # 随机选择基准避免最坏情况
    pivot_idx = random.randint(0, len(nums) - 1)
    pivot = nums[pivot_idx]

    # 三路分区：小于 | 等于 | 大于
    less = [x for x in nums if x < pivot]
    equal = [x for x in nums if x == pivot]
    greater = [x for x in nums if x > pivot]

    return quicksort(less) + equal + quicksort(greater)


def quicksort_inplace(nums: list[int], left: int = 0, right: int = None):
    """原地快排（面试推荐写法）"""
    if right is None:
        right = len(nums) - 1

    if left >= right:
        return

    # 随机基准
    pivot_idx = random.randint(left, right)
    nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]

    # 分区
    pivot = nums[right]
    store_idx = left

    for i in range(left, right):
        if nums[i] < pivot:
            nums[i], nums[store_idx] = nums[store_idx], nums[i]
            store_idx += 1

    nums[store_idx], nums[right] = nums[right], nums[store_idx]

    # 递归
    quicksort_inplace(nums, left, store_idx - 1)
    quicksort_inplace(nums, store_idx + 1, right)


if __name__ == "__main__":
    # 测试函数式快排
    assert quicksort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]
    assert quicksort([]) == []
    assert quicksort([1]) == [1]
    assert quicksort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    # 测试原地快排
    arr = [3, 6, 8, 10, 1, 2, 1]
    quicksort_inplace(arr)
    assert arr == [1, 1, 2, 3, 6, 8, 10]
    print("OK")
