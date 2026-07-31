# ---
# chapter: 1
# topic: Python 3.13+ 新 REPL
# section: 1.1.3
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 02_python_314_repl.py
# expected_runtime: <1s
# expected_output: REPL 新特性说明
# ---
# See: ../tutorial/01_Python编程基础.md (lines 89-109)
# Interview hooks:
#   1. Python 3.13 新 REPL 带来了哪些改进?
#   2. 什么是 bracketed paste 模式?
#   3. 多行编辑中 Alt+Enter 和 Esc+Enter 的区别?
"""
Python 3.13+ 新 REPL 特性（无需任何第三方库）
"""

# 1. 默认语法高亮 — 关键字、字符串、注释等自动着色
#    >>> def hello(name: str) -> str:
#    ...     return f"Hello, {name}!"
#    ...                    # ↑ 字符串高亮显示

# 2. 多行编辑 — 支持在历史和当前输入中跨行编辑
#    使用 Alt+Enter 或 Esc+Enter 插入新行，不再强制立即执行

# 3. 历史搜索增强 — 支持 Ctrl+R 反向搜索命令历史
#    (类似 bash/zsh 的 reverse-i-search)

# 4. 智能粘贴模式 — 粘贴多行代码时自动识别，避免逐行执行
#    （bracketed paste 支持）

# 5. 帮助文档直接显示 — help() 输出支持分页和语法高亮

print("Python 3.13+ 新 REPL 特性说明（注释文档）")

if __name__ == "__main__":
    print("OK")
