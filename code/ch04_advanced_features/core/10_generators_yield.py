# ---
# chapter: 4
# topic: Python高级特性与函数式编程
# section: 4.3.2 生成器函数 —— yield
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 10_generators_yield.py
# expected_runtime: <1s
# expected_output: 生成器基础 + 斐波那契 + yield from 展平
# ---
# See: ../tutorial/04_Python高级特性与函数式编程.md — §4.3.2 生成器函数 —— yield
# Interview hooks:
#   1. 生成器相对于普通函数的关键区别是什么？
#   2. yield 与 return 有什么不同？
#   3. yield from 的两个作用是什么？（委托 + 转发 send/throw）

"""
生成器（Generator）— 面试超高频考点

生成器是一种特殊的迭代器，使用 yield 关键字实现。
每次 yield 会暂停执行并返回值，下次从暂停处继续。

生成器的优势：
1. 惰性求值 —— 按需生成数据，节省内存
2. 代码简洁 —— 比手写迭代器类简单得多
3. 状态自动保存 —— yield 自动保存局部变量状态
"""

# ─────────────────────────────────────────────────────────────
# 基础生成器
# ─────────────────────────────────────────────────────────────


def count_up(n):
    """生成 1 到 n 的整数"""
    i = 1
    while i <= n:
        yield i  # 暂停，返回 i
        i += 1  # 下次从这里继续


# 使用生成器
gen = count_up(3)
print(type(gen))  # <class 'generator'>
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
# next(gen)               # StopIteration

# 用 for 循环遍历（自动处理 StopIteration）
for num in count_up(5):
    print(num, end=" ")  # 1 2 3 4 5
print()

# ─────────────────────────────────────────────────────────────
# 生成器的状态保存
# ─────────────────────────────────────────────────────────────


def fibonacci():
    """
    无限斐波那契数列生成器

    状态自动保存在生成器对象中：
    - a 和 b 的值在每次 yield 后自动保存
    - 下次从 yield 处继续执行
    """
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


fib = fibonacci()
for _ in range(10):
    print(next(fib), end=" ")  # 0 1 1 2 3 5 8 13 21 34
print()

# ─────────────────────────────────────────────────────────────
# 面试真题：用生成器实现大文件读取
# ─────────────────────────────────────────────────────────────


def read_large_file(filepath, chunk_size=1024):
    """
    用生成器逐块读取大文件 —— 避免内存溢出

    普通方式 f.read() 会将整个文件读入内存，
    生成器方式每次只读 chunk_size 字节到内存。
    """
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


# 逐行读取（更常用）
def read_lines(filepath):
    """逐行读取文件 —— yield 自动处理缓冲区"""
    with open(filepath, encoding="utf-8") as f:
        for line in f:  # 文件对象本身就是迭代器！
            yield line.strip()


# 处理大文件日志的实用生成器
def parse_log_file(filepath):
    """
    解析日志文件 —— 过滤 + 解析的生成器管道
    """
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # 跳过空行和注释
            # 解析日志行
            parts = line.split(" | ")
            if len(parts) >= 3:
                yield {
                    "timestamp": parts[0],
                    "level": parts[1],
                    "message": parts[2],
                }


# ─────────────────────────────────────────────────────────────
# 生成器表达式 —— 内存友好的推导式
# ─────────────────────────────────────────────────────────────

# 列表推导式 —— 立即生成所有数据，占用大量内存
squares_list = [x**2 for x in range(1000000)]  # 内存占用大

# 生成器表达式 —— 惰性求值，每次只生成一个
squares_gen = (x**2 for x in range(1000000))  # 几乎不占内存

# 可以直接用在需要迭代器的场景
total = sum(x**2 for x in range(1000000))  # 高效！

# 用 tempfile 演示 max(... open(...)) 模式 (避免依赖外部 file.txt)
import os
import tempfile

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
    tmp.write("alpha\nbeta\ngamma\ndelta")
    tmp_path = tmp.name
max_val = max(len(line) for line in open(tmp_path, encoding="utf-8"))
print(f"max line length: {max_val}")
os.unlink(tmp_path)

# ─────────────────────────────────────────────────────────────
# yield from —— 委托子生成器
# ─────────────────────────────────────────────────────────────


def sub_generator():
    """子生成器"""
    yield 1
    yield 2


def main_generator():
    """主生成器 —— 委托给子生成器"""
    yield "开始"
    yield from sub_generator()  # 等价于 for x in sub_generator(): yield x
    yield "结束"


print(list(main_generator()))  # ['开始', 1, 2, '结束']


# yield from 的核心用途：展平嵌套结构
def flatten(nested):
    """展平任意嵌套的列表"""
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)  # 递归委托
        else:
            yield item


nested = [1, [2, [3, 4], 5], 6, [7, 8]]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6, 7, 8]


if __name__ == "__main__":
    print("OK")
