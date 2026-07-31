# ---
# chapter: 7
# topic: Python 数据结构与算法
# section: 7.2.2 BFS 与 DFS
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: collections
# run: python 05_bfs_dfs.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/07_Python数据结构与算法.md#7.2.2-BFS-与-DFS
#
# Interview hooks:
#  1. 二叉树层序遍历（LeetCode 102）：BFS 用队列的层序如何记录每层节点？时间/空间复杂度？
#  2. 平衡二叉树（LeetCode 110）：自顶向下和自底向上递归的时间复杂度差别？O(n²) vs O(n)？
#  3. 图的 BFS/DFS：邻接表与邻接矩阵的存储复杂度？visited 集合是否必要？


from collections import deque


# ========== 通用二叉树节点 ==========
class TreeNode:
    """二叉树节点 — 本文件所有 BFS/DFS 题共用."""

    def __init__(self, val: int, left: "TreeNode | None" = None, right: "TreeNode | None" = None):
        self.val = val
        self.left = left
        self.right = right


# ========== 面试高频题：二叉树层序遍历 ==========
def level_order(root: TreeNode) -> list[list[int]]:
    """
    BFS 层序遍历

    时间复杂度: O(n)  空间复杂度: O(w)，w 为最大宽度
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result


# ========== 二叉树最大深度 ==========
def max_depth(root: TreeNode) -> int:
    """BFS 解法"""
    if not root:
        return 0

    depth = 0
    queue = deque([root])

    while queue:
        depth += 1
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return depth


def max_depth_dfs(root: TreeNode) -> int:
    """DFS 解法"""
    if not root:
        return 0
    return 1 + max(max_depth_dfs(root.left), max_depth_dfs(root.right))


# ========== 判断平衡二叉树 ==========
def is_balanced(root: TreeNode) -> bool:
    """
    自底向上的递归，避免重复计算
    时间复杂度: O(n)  空间复杂度: O(h)
    """

    def check(node):
        if not node:
            return 0  # 空节点高度为0

        left = check(node.left)
        if left == -1:
            return -1  # 左子树不平衡

        right = check(node.right)
        if right == -1:
            return -1  # 右子树不平衡

        if abs(left - right) > 1:
            return -1  # 当前节点不平衡

        return max(left, right) + 1

    return check(root) != -1


# ========== 二叉搜索树验证 ==========
def is_valid_bst(root: TreeNode) -> bool:
    """
    利用 BST 中序遍历有序的性质
    时间复杂度: O(n)  空间复杂度: O(h)
    """

    def inorder(node):
        if not node:
            return True

        if not inorder(node.left):
            return False

        # 检查当前值是否大于前一个值
        if node.val <= inorder.prev:
            return False
        inorder.prev = node.val

        return inorder(node.right)

    inorder.prev = float("-inf")
    return inorder(root)


# ========== 图的 BFS/DFS ==========
from collections import defaultdict


class Graph:
    """邻接表表示的有向图"""

    def __init__(self):
        self.adj = defaultdict(list)

    def add_edge(self, u: int, v: int):
        self.adj[u].append(v)

    def bfs(self, start: int) -> list[int]:
        """图的 BFS"""
        visited = set([start])
        queue = deque([start])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self.adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return result

    def dfs(self, start: int) -> list[int]:
        """图的 DFS（迭代）"""
        visited = set()
        stack = [start]
        result = []

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                result.append(node)
                for neighbor in reversed(self.adj[node]):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return result

    def has_path(self, start: int, end: int) -> bool:
        """判断是否存在从 start 到 end 的路径"""
        if start == end:
            return True

        visited = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in self.adj[node]:
                if neighbor == end:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False


if __name__ == "__main__":
    # 构造测试树:
    #       3
    #      / \
    #     9   20
    #        /  \
    #       15   7
    root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))

    # 层序遍历
    assert level_order(root) == [[3], [9, 20], [15, 7]]

    # 最大深度
    assert max_depth(root) == 3
    assert max_depth_dfs(root) == 3

    # 平衡二叉树
    assert is_balanced(root) is True
    unbalanced = TreeNode(1, TreeNode(2, TreeNode(3, TreeNode(4), TreeNode(4)), TreeNode(3)), TreeNode(2))
    assert is_balanced(unbalanced) is False

    # BST 验证
    bst = TreeNode(2, TreeNode(1), TreeNode(3))
    assert is_valid_bst(bst) is True
    invalid_bst = TreeNode(1, TreeNode(2), TreeNode(3))
    assert is_valid_bst(invalid_bst) is False

    # 图 BFS/DFS
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(2, 3)
    g.add_edge(3, 3)
    assert g.bfs(2)[0] == 2
    assert g.has_path(1, 3) is True
    assert g.has_path(3, 1) is False
    print("OK")
