# ---
# chapter: 1
# topic: PEP 8 — Python 代码风格指南
# section: 1.5.2
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 21_pep8_style.py
# expected_runtime: <1s
# expected_output: PEP 8 风格示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 1319-1369)
# Interview hooks:
#   1. PEP 8 中模块/类/函数/常量的命名规范?
#   2. Python 类型注解 Optional[T] 与 Union[T, None] 的关系?
#   3. 文档字符串 (docstring) 的 Google/NumPy/Sphinx 风格区别?
"""
PEP 8 — Python 代码风格指南（面试可能问到）

核心规则：
1. 缩进：4 个空格（不用 Tab）
2. 行宽：最大 79 字符（文档字符串 72 字符）
3. 命名：
   - 模块/包：小写 + 下划线（my_module）
   - 类：驼峰命名（MyClass）
   - 函数/变量：小写 + 下划线（my_function）
   - 常量：全大写（MAX_SIZE）
   - 私有：前导下划线（_private_var）
4. 空行：函数间 2 行，类内方法间 1 行
5. import：标准库 → 第三方 → 本地，每组空一行
"""


# 文档字符串规范
def calculate_area(length: float, width: float) -> float:
    """计算矩形面积。

    Args:
        length: 矩形的长度，必须为正数。
        width: 矩形的宽度，必须为正数。

    Returns:
        矩形的面积。

    Raises:
        ValueError: 如果 length 或 width 为负数。

    Examples:
        >>> calculate_area(3.0, 4.0)
        12.0
    """
    if length < 0 or width < 0:
        raise ValueError("长度和宽度必须为正数")
    return length * width


# 演示
print(f"面积: {calculate_area(3.0, 4.0)}")

# 类型注解（Python 3.5+，大型项目推荐）


def process_data(items: list[int], config: dict[str, str | int] | None = None) -> list[str]:
    """带类型注解的函数"""
    if config is None:
        config = {}
    return [str(item) for item in items]


# 演示
print(f"处理结果: {process_data([1, 2, 3])}")
print(f"带配置: {process_data([1, 2], {'mode': 'fast'})}")

if __name__ == "__main__":
    print("OK")
