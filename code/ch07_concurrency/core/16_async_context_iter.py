# ---
# chapter: 7
# topic: Python 并发编程
# topic_id: concurrency.async_context_iter
# difficulty: ⭐⭐⭐
# tier: core
# deps: [aiosqlite]
# run: python 16_async_context_iter.py
# expected_runtime: ~0.5s
# expected_output: prints 0..4 from the async iterator
# ---
# See: ../../../07_Python并发编程.md
# Interview hooks:
#   - __aenter__/__aexit__ 与 __enter__/__exit__ 的对应关系？
#   - 异步迭代器协议的方法名？与同步迭代器有何不同？
#   - async for 与 async with 在底层事件循环中的协作方式？
import asyncio


# ========== 异步上下文管理器 ==========
class AsyncDatabase:
    """异步数据库连接上下文管理器"""

    async def __aenter__(self):
        # 这里仅作演示，不真正连接
        self.conn = {"opened": True, "name": "test.db"}
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.conn = None


async def use_async_context():
    async with AsyncDatabase() as conn:
        print(f"打开连接: {conn}")
        # await conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT)")
        # await conn.commit()


# ========== 异步迭代器 ==========
class AsyncRange:
    """异步范围迭代器"""

    def __init__(self, start, end, delay=0.05):
        self.current = start
        self.end = end
        self.delay = delay

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current >= self.end:
            raise StopAsyncIteration
        await asyncio.sleep(self.delay)
        value = self.current
        self.current += 1
        return value


async def use_async_iter():
    async for i in AsyncRange(0, 5):
        print(f"异步迭代: {i}")


async def main():
    await use_async_context()
    print("---")
    await use_async_iter()


if __name__ == "__main__":
    asyncio.run(main())
    print("OK")
