# ---
# chapter: 7
# topic: Python 数据结构与算法
# section: 7.4.2 双指针
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 09_two_pointers.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/07_Python数据结构与算法.md#7.4.2-双指针
#
# Interview hooks:
#  1. 盛最多水的容器（LeetCode 11）：为什么每次移动较短的线段就能找到最大值？反证法证明？
#  2. 三数之和（LeetCode 15）：去重的三处：i 跳过重复、left 跳过重复、right 跳过重复？剪枝条件？
#  3. 双指针的适用条件：为什么必须先排序？双指针在链表合并中的应用（LeetCode 21）？


# ========== 面试高频题：盛最多水的容器 ==========
def max_area(heights: list[int]) -> int:
    """
    双指针从两端向中间移动

    时间复杂度: O(n)  空间复杂度: O(1)

    策略：每次移动较短的那根线，因为面积受限于短线
    """
    left, right = 0, len(heights) - 1
    max_water = 0

    while left < right:
        width = right - left
        height = min(heights[left], heights[right])
        max_water = max(max_water, width * height)

        # 移动较短的那边
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1

    return max_water


# ========== 面试高频题：三数之和 ==========
def three_sum(nums: list[int]) -> list[list[int]]:
    """
    排序 + 双指针

    时间复杂度: O(n²)  空间复杂度: O(1)（不含结果）
    """
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n - 2):
        # 去重：跳过重复的第一个数
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # 剪枝
        if nums[i] > 0:
            break

        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                # 去重
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result


if __name__ == "__main__":
    # 盛最多水的容器
    assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert max_area([1, 1]) == 1
    assert max_area([4, 3, 2, 1, 4]) == 16
    assert max_area([1, 2, 1]) == 2

    # 三数之和
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
    assert three_sum([0, 1, 1]) == []
    assert three_sum([0, 0, 0]) == [[0, 0, 0]]
    assert three_sum([]) == []
    print("OK")
