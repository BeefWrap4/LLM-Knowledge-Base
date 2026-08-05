# ---
# chapter: 4
# topic: Python 迭代协议与函数式编程
# topic_id: iteration_functional.context_managers
# difficulty: ⭐⭐⭐
# tier: core
# deps: contextlib
# run: python 12_context_managers.py
# expected_runtime: <1s
# expected_output: 类 + @contextmanager 两种实现
# ---
# See: ../../../04_Python迭代协议与函数式编程.md
# Interview hooks:
#   1. with 语句背后的 __enter__/__exit__ 协议如何工作？
#   2. @contextmanager 装饰器是如何把生成器转为上下文管理器的？
#   3. __exit__ 返回 True/False 各代表什么含义？

"""
上下文管理器（Context Manager）— with 语句的底层机制

协议：实现 __enter__() 和 __exit__() 两个方法
"""

# ─────────────────────────────────────────────────────────────
# 类实现上下文管理器
# ─────────────────────────────────────────────────────────────


class DatabaseConnection:
    """
    数据库连接上下文管理器
    """

    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None

    def __enter__(self):
        """进入 with 块时调用 —— 获取资源"""
        print(f"🔗 连接数据库: {self.connection_string}")
        self.connection = f"<连接: {self.connection_string}>"
        return self  # 返回的资源会被 as 变量接收

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 块时调用 —— 释放资源"""
        print("🔒 关闭数据库连接")
        self.connection = None

        if exc_type:
            print(f"  异常类型: {exc_type.__name__}")
            print(f"  异常信息: {exc_val}")
            # 返回 True 表示异常已被处理，不向外传播
            # 返回 False 表示异常继续传播
            return False

    def query(self, sql):
        print(f"执行 SQL: {sql}")
        return f"[{sql}] 的结果"


# 使用
with DatabaseConnection("mysql://localhost/mydb") as db:
    result = db.query("SELECT * FROM users")
    print(result)
# 🔗 连接数据库: mysql://localhost/mydb
# 执行 SQL: SELECT * FROM users
# [SELECT * FROM users] 的结果
# 🔒 关闭数据库连接

# ─────────────────────────────────────────────────────────────
# @contextmanager 装饰器（更简洁）
# ─────────────────────────────────────────────────────────────

from contextlib import contextmanager


@contextmanager
def db_connection(connection_string):
    """
    用生成器实现上下文管理器 —— 更 Pythonic

    yield 之前的代码 = __enter__
    yield 返回值     = __enter__ 的返回值
    yield 之后的代码 = __exit__（无论是否异常都会执行）
    """
    # ── __enter__ 部分 ──
    print(f"🔗 连接数据库: {connection_string}")
    connection = f"<连接: {connection_string}>"

    try:
        yield connection  # ← 这行相当于 return，控制权交给 with 块
    finally:
        # ── __exit__ 部分（finally 保证一定执行）──
        print("🔒 关闭数据库连接")


# 使用
with db_connection("postgresql://localhost/prod") as conn:
    print(f"使用连接: {conn}")

# ─────────────────────────────────────────────────────────────
# 多个上下文管理器
# ─────────────────────────────────────────────────────────────


@contextmanager
def file_reader(filepath):
    f = open(filepath)
    try:
        yield f
    finally:
        f.close()


@contextmanager
def timer_ctx(name):
    import time

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"⏱ {name}: {elapsed:.4f}s")


# Python 3.10+ 支持括号语法
# with (
#     file_reader("input.txt") as fin,
#     file_reader("output.txt") as fout,
#     timer_ctx("文件拷贝"):
# ):
#     fout.write(fin.read())

# Python 3.9 及以下
# with file_reader("input.txt") as fin, \
#      file_reader("output.txt") as fout, \
#      timer_ctx("文件拷贝"):
#     fout.write(fin.read())


if __name__ == "__main__":
    print("OK")
