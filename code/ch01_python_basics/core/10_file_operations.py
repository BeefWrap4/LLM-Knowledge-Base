# ---
# chapter: 1
# topic: 文件操作最佳实践
# section: 1.2.4
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 10_file_operations.py
# expected_runtime: <1s
# expected_output: 文件操作示例（无文件时跳过/报错）
# ---
# See: ../tutorial/01_Python编程基础.md (lines 432-472)
# Interview hooks:
#   1. with 语句的底层原理(__enter__/__exit__)?
#   2. 大文件读取时如何避免内存溢出?
#   3. 文件模式 "r"/"w"/"a"/"x"/"b"/"+" 的区别?
"""
文件操作最佳实践
"""

import tempfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# with 语句 — 自动关闭文件（面试必考）
# ─────────────────────────────────────────────────────────────


# ✅ 正确写法：with 语句确保资源释放
def read_file_safe(filepath: str) -> str:
    """安全读取文件内容"""
    try:
        with open(filepath, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except UnicodeDecodeError:
        # 尝试其他编码
        with open(filepath, encoding="gbk") as f:
            return f.read()


# 大文件读取：逐行读取（避免内存溢出）
def read_large_file(filepath: str):
    """逐行读取大文件，内存友好"""
    with open(filepath, encoding="utf-8") as f:
        for line in f:  # 每次只读一行到内存
            yield line.strip()  # yield 使函数成为生成器


# ─────────────────────────────────────────────────────────────
# 文件模式速查表
# ─────────────────────────────────────────────────────────────
# 模式    说明
# ─────────────────
# "r"     只读（默认）
# "w"     只写，文件存在则清空
# "a"     追加写入
# "x"     独占创建，文件存在则报错
# "b"     二进制模式（如 "rb"）
# "+"     读写模式（如 "r+"）

def demonstrate_file_operations() -> None:
    """在自动清理的临时目录中完成写入与读取，不污染当前工作目录。"""
    with tempfile.TemporaryDirectory(prefix="python-file-demo-") as temp_dir:
        demo_file = Path(temp_dir) / "demo.txt"
        with demo_file.open("w", encoding="utf-8") as file_handle:
            file_handle.write("line1\nline2\nline3\n")

        content = read_file_safe(str(demo_file))
        print(f"文件内容: {content!r}")

        lines = list(read_large_file(str(demo_file)))
        print(f"逐行读取: {lines}")


if __name__ == "__main__":
    demonstrate_file_operations()
    print("OK")
