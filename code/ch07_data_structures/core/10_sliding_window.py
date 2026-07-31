# ---
# chapter: 7
# topic: Python 数据结构与算法
# section: 7.4.3 滑动窗口
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: collections
# run: python 10_sliding_window.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/07_Python数据结构与算法.md#7.4.3-滑动窗口
#
# Interview hooks:
#  1. 无重复字符的最长子串（LeetCode 3）：为什么滑动窗口 + set 是 O(n)？左右指针的移动规则？
#  2. 最小覆盖子串（LeetCode 76）：missing 变量的意义？为什么是 O(|s| + |t|)？
#  3. 滑动窗口的两种类型：可变窗口（求最长/最小）和固定窗口（求所有异位词）？模板差异？


# ========== 面试高频题：无重复字符的最长子串 ==========
def length_of_longest_substring(s: str) -> int:
    """
    滑动窗口 + 哈希集合

    时间复杂度: O(n)  空间复杂度: O(min(m, n))
    m 为字符集大小
    """
    char_set = set()
    left = max_len = 0

    for right in range(len(s)):
        # 右指针扩展窗口
        while s[right] in char_set:
            # 左指针收缩窗口直到无重复
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len


# ========== 面试高频题：最小覆盖子串 ==========
def min_window(s: str, t: str) -> str:
    """
    滑动窗口：找到 s 中包含 t 所有字符的最小子串

    时间复杂度: O(|s| + |t|)  空间复杂度: O(|t|)
    """
    from collections import Counter

    need = Counter(t)  # t 中每个字符需要的次数
    missing = len(t)  # 还缺少的字符总数

    left = start = end = 0

    for right, char in enumerate(s, 1):
        # 扩展右边界
        if need[char] > 0:
            missing -= 1
        need[char] -= 1

        # 窗口已覆盖 t，尝试收缩左边界
        while missing == 0:
            if end == 0 or right - left < end - start:
                start, end = left, right

            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1

    return s[start:end]


# ========== 面试高频题：找到字符串中所有字母异位词 ==========
def find_anagrams(s: str, p: str) -> list[int]:
    """
    固定窗口大小的滑动窗口

    时间复杂度: O(n)  空间复杂度: O(1)（字符集大小固定26）
    """
    from collections import Counter

    if len(p) > len(s):
        return []

    p_count = Counter(p)
    window_count = Counter(s[: len(p) - 1])
    result = []

    for i in range(len(p) - 1, len(s)):
        window_count[s[i]] += 1  # 右边加入

        if window_count == p_count:
            result.append(i - len(p) + 1)

        # 左边移除
        left_char = s[i - len(p) + 1]
        window_count[left_char] -= 1
        if window_count[left_char] == 0:
            del window_count[left_char]

    return result


if __name__ == "__main__":
    # 无重复字符的最长子串
    assert length_of_longest_substring("abcabcbb") == 3
    assert length_of_longest_substring("bbbbb") == 1
    assert length_of_longest_substring("pwwkew") == 3
    assert length_of_longest_substring("") == 0
    assert length_of_longest_substring(" ") == 1

    # 最小覆盖子串
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window("a", "a") == "a"
    assert min_window("a", "aa") == ""
    assert min_window("ab", "b") == "b"

    # 找到字符串中所有字母异位词
    assert find_anagrams("cbaebabacd", "abc") == [0, 6]
    assert find_anagrams("abab", "ab") == [0, 1, 2]
    assert find_anagrams("a", "ab") == []
    print("OK")
