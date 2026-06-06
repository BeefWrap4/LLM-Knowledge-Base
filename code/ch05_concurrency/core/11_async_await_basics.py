# ---
# chapter: 05
# topic: Python并发编程
# section: 5.5.1 async/await 基础
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 11_async_await_basics.py
# expected_runtime: ~4s
# expected_output: serial + create_task + gather demos
# ---
# See: ../tutorial/05_Python并发编程.md#551-asyncawait-基础
# Interview hooks:
#   - async def 和普通 def 的区别？调用 async 函数会怎样？
#   - await 后面能接哪些对象？
#   - asyncio.gather 和 asyncio.create_task 的取舍？
import asyncio


async def say_hello(name, delay):
    """async def 定义协程函数"""
    print(f"Hello {name}, 等待 {delay}s...")
    await asyncio.sleep(delay)  # await 挂起当前协程，让出执行权
    print(f"Goodbye {name}")
    return f"{name} 完成"


async def main():
    """入口协程"""
    # await 直接等待一个协程完成
    result = await say_hello("Alice", 1)
    print(f"结果: {result}\n")

    # ========== 并行执行多个协程 ==========
    # 方式1：create_task 创建后台任务
    task1 = asyncio.create_task(say_hello("Bob", 2))
    task2 = asyncio.create_task(say_hello("Carol", 1))

    # 此时两个任务已经在后台运行
    result1 = await task1
    result2 = await task2
    print(f"Task1: {result1}, Task2: {result2}\n")

    # 方式2：gather 等待所有协程完成（更简洁）
    results = await asyncio.gather(
        say_hello("Dave", 1),
        say_hello("Eve", 2),
        say_hello("Frank", 1),
    )
    print(f"Gather 结果: {results}")


if __name__ == "__main__":
    # 启动事件循环
    asyncio.run(main())
    print("OK")
