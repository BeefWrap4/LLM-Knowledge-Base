# ---
# chapter: 7
# topic: Python 数据结构与算法
# section: 7.5.2 经典 DP 题目
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: bisect
# run: python 11_dynamic_programming.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/07_Python数据结构与算法.md#7.5.2-经典-DP-题目
#
# Interview hooks:
#  1. 爬楼梯（LeetCode 70）：为什么是斐波那契数列？滚动数组如何把空间从 O(n) 优化到 O(1)？
#  2. 零钱兑换（LeetCode 322）：完全背包问题？为什么 dp 数组初始化为 inf，dp[0] = 0？
#  3. 编辑距离（LeetCode 72）：三种操作（插入、删除、替换）的状态转移如何推导？二维 DP 如何优化为一维？


# ========== 爬楼梯 ==========
def climb_stairs(n: int) -> int:
    """
    每次可以爬 1 或 2 阶

    状态: dp[i] = 到达第 i 阶的方法数
    转移: dp[i] = dp[i-1] + dp[i-2]

    时间复杂度: O(n)  空间复杂度: O(1)（滚动数组优化）
    """
    if n <= 2:
        return n

    prev2, prev1 = 1, 2  # dp[i-2], dp[i-1]

    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1


# ========== 零钱兑换 ==========
def coin_change(coins: list[int], amount: int) -> int:
    """
    凑成 amount 的最少硬币数

    状态: dp[i] = 凑成金额 i 的最少硬币数
    转移: dp[i] = min(dp[i - coin] + 1) for coin in coins

    时间复杂度: O(n * amount)  空间复杂度: O(amount)
    """
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0  # 凑成 0 需要 0 个硬币

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float("inf") else -1


# ========== 最长递增子序列 (LIS) ==========
def length_of_lis(nums: list[int]) -> int:
    """
    时间复杂度: O(n²) — 基础 DP 版本
    空间复杂度: O(n)
    """
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n  # dp[i] = 以 nums[i] 结尾的最长递增子序列长度

    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def length_of_lis_binary(nums: list[int]) -> int:
    """
    二分优化版本

    时间复杂度: O(nlogn)  空间复杂度: O(n)

    tails[i] = 长度为 i+1 的递增子序列的最小尾部元素
    """
    import bisect

    tails = []
    for num in nums:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num

    return len(tails)


# ========== 最长公共子序列 (LCS) ==========
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    时间复杂度: O(m*n)  空间复杂度: O(m*n)，可优化至 O(min(m,n))

    状态: dp[i][j] = text1[0:i] 和 text2[0:j] 的 LCS 长度
    转移:
      - 如果 text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
      - 否则: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# ========== 0/1 背包问题 ==========
def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    """
    0/1 背包：每件物品只能选一次

    状态: dp[i][w] = 前 i 件物品，容量 w 时的最大价值
    转移:
      - 不选第 i 件: dp[i][w] = dp[i-1][w]
      - 选第 i 件: dp[i][w] = dp[i-1][w-weights[i]] + values[i]

    空间优化：一维数组倒序遍历
    """
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        # 倒序遍历！避免重复选择
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]


# ========== 编辑距离 ==========
def min_distance(word1: str, word2: str) -> int:
    """
    将 word1 转换成 word2 的最少操作数（插入、删除、替换）

    状态: dp[i][j] = word1[0:i] -> word2[0:j] 的最小操作数
    时间复杂度: O(m*n)  空间复杂度: O(m*n)
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界
    for i in range(m + 1):
        dp[i][0] = i  # word1 -> 空字符串，删除 i 次
    for j in range(n + 1):
        dp[0][j] = j  # 空字符串 -> word2，插入 j 次

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 字符相同，无需操作
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # 删除 word1[i-1]
                    dp[i][j - 1] + 1,  # 插入 word2[j-1]
                    dp[i - 1][j - 1] + 1,  # 替换
                )

    return dp[m][n]


if __name__ == "__main__":
    # 爬楼梯
    assert climb_stairs(2) == 2
    assert climb_stairs(3) == 3
    assert climb_stairs(5) == 8

    # 零钱兑换
    assert coin_change([1, 5, 10, 25], 41) == 4  # 25+10+5+1
    assert coin_change([2], 3) == -1
    assert coin_change([1], 0) == 0

    # 最长递增子序列
    assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert length_of_lis([0, 1, 0, 3, 2, 3]) == 4
    assert length_of_lis_binary([10, 9, 2, 5, 3, 7, 101, 18]) == 4

    # 最长公共子序列
    assert longest_common_subsequence("abcde", "ace") == 3
    assert longest_common_subsequence("abc", "abc") == 3
    assert longest_common_subsequence("abc", "def") == 0

    # 0/1 背包
    # weights=[2,3,4,5], values=[3,4,5,6], capacity=8
    #   最佳: 5+3=8 → value 6+4=10
    assert knapsack([2, 3, 4, 5], [3, 4, 5, 6], 8) == 10
    # weights=[1,2,3], values=[6,10,12], capacity=5
    #   最佳: 2+3=5 → value 10+12=22
    assert knapsack([1, 2, 3], [6, 10, 12], 5) == 22

    # 编辑距离
    assert min_distance("horse", "ros") == 3
    assert min_distance("intention", "execution") == 5
    assert min_distance("", "abc") == 3
    assert min_distance("abc", "") == 3
    print("OK")
