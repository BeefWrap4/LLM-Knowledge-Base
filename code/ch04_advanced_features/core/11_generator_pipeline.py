# ---
# chapter: 4
# topic: Python高级特性与函数式编程
# section: 4.3.3 生成器实现大文件处理
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: collections
# run: python 11_generator_pipeline.py
# expected_runtime: <1s
# expected_output: 演示生成器管道函数（不实际执行 I/O）
# ---
# See: ../tutorial/04_Python高级特性与函数式编程.md — §4.3.3 生成器实现大文件处理
# Interview hooks:
#   1. 生成器管道（Pipeline）模式为什么能节省内存？
#   2. readlines() 和生成器逐行读取在内存占用上有什么差异？
#   3. 解释惰性求值（lazy evaluation）的好处。

"""
生成器实现大文件处理 —— 面试常考场景

核心思路：用生成器构建数据处理管道（Pipeline），
         数据像流水一样经过多个处理阶段，
         每个阶段只处理当前数据块，不加载全部数据。
"""


# ─────────────────────────────────────────────────────────────
# 数据处理管道模式
# ─────────────────────────────────────────────────────────────


def read_chunks(filepath, chunk_size=8192):
    """阶段1：读取文件块"""
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def decode_lines(chunks):
    """阶段2：将字节块解码为文本行"""
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            yield line.decode("utf-8")
    if buffer:
        yield buffer.decode("utf-8")


def filter_lines(lines, keyword):
    """阶段3：过滤包含关键词的行"""
    for line in lines:
        if keyword in line:
            yield line


def parse_records(lines):
    """阶段4：解析为结构化数据"""
    for line in lines:
        parts = line.split(",")
        if len(parts) >= 3:
            yield {
                "id": parts[0].strip(),
                "name": parts[1].strip(),
                "value": float(parts[2].strip()),
            }


# 组合管道（惰性执行，不占用大量内存）
def process_file_pipeline(filepath, keyword):
    """完整的数据处理管道"""
    chunks = read_chunks(filepath)
    lines = decode_lines(chunks)
    filtered = filter_lines(lines, keyword)
    records = parse_records(filtered)
    return records  # 返回生成器，尚未执行任何处理！


# ─────────────────────────────────────────────────────────────
# 实际应用：逐行读取 + 统计
# ─────────────────────────────────────────────────────────────


def line_count(filepath):
    """统计行数 —— 不加载整个文件"""
    with open(filepath, encoding="utf-8") as f:
        return sum(1 for _ in f)  # 生成器表达式 + sum


def grep_generator(pattern, filepath):
    """实现 grep 功能 —— 返回匹配行的生成器"""
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if pattern in line:
                yield i, line.strip()


# 统计每个 IP 的访问次数（类似 awk）
def count_ip_frequency(logfile):
    """统计日志中每个 IP 的出现次数 —— 流式处理"""
    from collections import Counter

    def extract_ips(filepath):
        with open(filepath) as f:
            for line in f:
                # 假设日志格式: "IP - - [timestamp] ..."
                parts = line.split()
                if parts:
                    yield parts[0]  # 第一个字段是 IP

    return Counter(extract_ips(logfile))


if __name__ == "__main__":
    # 单元测试：演示管道函数都是生成器工厂
    print("process_file_pipeline is callable:", callable(process_file_pipeline))
    print("OK")
