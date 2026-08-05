# ---
# chapter: 8
# topic: Python 数据结构与算法
# topic_id: data_structures.binary_search
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 08_binary_search.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../../../08_Python数据结构与算法.md
#
# Interview hooks:
#  1. 二分查找的标准模板：为什么 mid = left + (right - left) // 2 而不是 (left + right) // 2？防止整数溢出？
#  2. 边界条件：while left <= right vs while left < right 的差别？什么场景用哪种？
#  3. 旋转排序数组：如何判断 target 在哪半边？nums[mid] >= nums[left] 与 nums[mid] <= nums[right] 的判定？


def binary_search(nums: list[int], target: int) -> int:
    """
    二分查找（标准模板）

    时间复杂度: O(logn)  空间复杂度: O(1)
    前提：数组有序

    返回 target 的索引，不存在返回 -1
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2  # 避免溢出

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# ========== 变体：查找第一个/最后一个等于 target 的位置 ==========
def find_first(nums: list[int], target: int) -> int:
    """查找第一个等于 target 的索引"""
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            result = mid
            right = mid - 1  # 继续在左边找
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result


def find_last(nums: list[int], target: int) -> int:
    """查找最后一个等于 target 的索引"""
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            result = mid
            left = mid + 1  # 继续在右边找
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result


# ========== 变体：查找旋转排序数组的最小值 ==========
def find_min_rotated(nums: list[int]) -> int:
    """
    旋转排序数组找最小值

    时间复杂度: O(logn)  空间复杂度: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] > nums[right]:
            # 最小值在右半部分
            left = mid + 1
        elif nums[mid] < nums[right]:
            # 最小值在左半部分（含 mid）
            right = mid
        else:
            # nums[mid] == nums[right]，无法判断，缩小范围
            right -= 1

    return nums[left]


if __name__ == "__main__":
    # 标准二分查找
    assert binary_search([1, 3, 5, 7, 9, 11, 13], 7) == 3
    assert binary_search([1, 3, 5, 7, 9, 11, 13], 1) == 0
    assert binary_search([1, 3, 5, 7, 9, 11, 13], 13) == 6
    assert binary_search([1, 3, 5, 7, 9, 11, 13], 0) == -1
    assert binary_search([], 5) == -1

    # 找第一个/最后一个
    nums = [1, 2, 2, 2, 3, 4, 5]
    assert find_first(nums, 2) == 1
    assert find_last(nums, 2) == 3
    assert find_first(nums, 6) == -1

    # 旋转排序数组找最小值
    assert find_min_rotated([3, 4, 5, 1, 2]) == 1
    assert find_min_rotated([4, 5, 6, 7, 0, 1, 2]) == 0
    assert find_min_rotated([11, 13, 15, 17]) == 11
    assert find_min_rotated([2, 1]) == 1
    print("OK")
