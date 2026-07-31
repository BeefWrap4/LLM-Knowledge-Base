# ---
# chapter: 05
# topic: Python并发编程
# section: 5.5.4 同步阻塞代码如何放入异步
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 14_sync_blocking_in_async.py
# expected_runtime: ~4s
# expected_output: 4 demos of run_in_executor / asyncio.to_thread / ProcessPool
# ---
# See: ../tutorial/05_Python并发编程.md#554--面试高频题同步阻塞代码如何放入异步
# Interview hooks:
#   - 为什么在协程中直接调用 time.sleep() 会阻塞整个事件循环？
#   - loop.run_in_executor(None, ...) 第一个参数传 None 含义？
#   - asyncio.to_thread 与 run_in_executor 的关系？
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def blocking_io(filename):
    """
    同步阻塞函数（如文件读写、数据库查询、CPU 计算）
    不能直接 await！
    """
    time.sleep(0.5)  # 模拟阻塞操作
    return f"{filename} 读取完成"


def cpu_bound_task(n):
    """CPU 密集型任务"""
    count = 0
    for i in range(n):
        count += i * i
    return count


async def main():
    # ========== 错误示范：在协程中直接调用阻塞函数 ==========
    # result = blocking_io("data.txt")  # ❌ 会阻塞整个事件循环！

    # ========== 正确方案1：run_in_executor（线程池）==========
    loop = asyncio.get_running_loop()

    # IO 密集型阻塞操作 → 线程池
    result = await loop.run_in_executor(
        None,  # None 使用默认线程池
        blocking_io,  # 同步函数
        "data.txt",  # 函数参数
    )
    print(f"方案1 结果: {result}")

    # ========== 正确方案2：asyncio.to_thread（Python 3.9+）==========
    result2 = await asyncio.to_thread(blocking_io, "data.txt")
    print(f"方案2 结果: {result2}")

    # ========== 正确方案3：ProcessPoolExecutor（CPU 密集型）==========
    with ThreadPoolExecutor() as pool:  # IO 用线程池
        result3 = await loop.run_in_executor(pool, blocking_io, "data.txt")
        print(f"方案3 结果: {result3}")

    # ========== CPU 密集型 → 进程池 ==========
    with ProcessPoolExecutor() as pool:
        result4 = await loop.run_in_executor(pool, cpu_bound_task, 1000000)
        print(f"CPU 任务结果: {result4}")


if __name__ == "__main__":
    asyncio.run(main())
    print("OK")
