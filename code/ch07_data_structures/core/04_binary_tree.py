# ---
# chapter: 7
# topic: Python 数据结构与算法
# section: 7.2.1 二叉树基础
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 04_binary_tree.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/07_Python数据结构与算法.md#7.2.1-二叉树基础
#
# Interview hooks:
#  1. 二叉树的三种 DFS 遍历（LeetCode 144/94/145）：递归和迭代实现？后序遍历的迭代为什么要用"根右左"再反转？
#  2. 中序遍历的应用：为什么 BST 的中序遍历是有序的？如何用迭代中序判断 BST？
#  3. 递归 vs 迭代：栈模拟递归的通用写法？两种方法的时间和空间复杂度？


class TreeNode:
    """二叉树节点"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# ========== 前序遍历（递归 + 迭代） ==========
def preorder_recursive(root: TreeNode) -> list[int]:
    """递归前序：根 -> 左 -> 右"""
    result = []
    def dfs(node):
        if not node:
            return
        result.append(node.val)  # 访问根
        dfs(node.left)           # 遍历左
        dfs(node.right)          # 遍历右
    dfs(root)
    return result


def preorder_iterative(root: TreeNode) -> list[int]:
    """迭代前序：使用栈模拟递归"""
    if not root:
        return []

    result, stack = [], [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        # 先压右再压左，保证左先出
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result


# ========== 中序遍历（递归 + 迭代） ==========
def inorder_iterative(root: TreeNode) -> list[int]:
    """迭代中序：左 -> 根 -> 右"""
    result, stack = [], []
    curr = root

    while curr or stack:
        # 走到最左边
        while curr:
            stack.append(curr)
            curr = curr.left
        # 弹出访问，然后转向右
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right

    return result


# ========== 后序遍历（迭代） ==========
def postorder_iterative(root: TreeNode) -> list[int]:
    """迭代后序：左 -> 右 -> 根

    技巧：前序(根左右)的变体(根右左)再反转
    """
    if not root:
        return []

    result, stack = [], [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return result[::-1]  # 反转得到后序


if __name__ == "__main__":
    # 构造测试树:
    #       1
    #      / \
    #     2   3
    #    / \ / \
    #   4  5 6  7
    root = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3, TreeNode(6), TreeNode(7))
    )

    # 前序
    assert preorder_recursive(root) == [1, 2, 4, 5, 3, 6, 7]
    assert preorder_iterative(root) == [1, 2, 4, 5, 3, 6, 7]

    # 中序
    assert inorder_iterative(root) == [4, 2, 5, 1, 6, 3, 7]

    # 后序
    assert postorder_iterative(root) == [4, 5, 2, 6, 7, 3, 1]

    # 空树
    assert preorder_recursive(None) == []
    assert preorder_iterative(None) == []
    assert inorder_iterative(None) == []
    assert postorder_iterative(None) == []

