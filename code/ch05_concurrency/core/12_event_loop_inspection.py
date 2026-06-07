# ---
# chapter: 05
# topic: Python并发编程
# section: 5.5.2 事件循环原理
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 12_event_loop_inspection.py
# expected_runtime: <1s
# expected_output: prints current event loop info, runs a tiny coroutine manually
# ---
# See: ../tutorial/05_Python并发编程.md#552-事件循环原理
# Interview hooks:
#   - 事件循环的工作流程是怎样的？Task 何时被调度？
#   - asyncio.run / run_until_complete / new_event_loop 三者关系？
#   - Selector 是什么？epoll / kqueue / IOCP 的关系？
import asyncio

# 查看和获取事件循环
loop = asyncio.get_event_loop()
print(f"当前事件循环: {loop}")


async def tiny_coro():
    await asyncio.sleep(0)
    return "done"


# asyncio.run() 的底层等价实现
def run_coroutine(coroutine):
    """等价于 asyncio.run()"""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


result = run_coroutine(tiny_coro())
print(f"手动事件循环结果: {result}")


async def main():
    # 触发 get_event_loop 的 deprecation 时使用 get_running_loop
    running = asyncio.get_running_loop()
    print(f"运行中事件循环: {running}")


if __name__ == "__main__":
    asyncio.run(main())
