# ---
# chapter: 05
# topic: Python并发编程
# section: 5.5.3 create_task vs gather vs TaskGroup
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 13_create_task_gather_taskgroup.py
# expected_runtime: ~3s
# expected_output: serial create_task, gather with return_exceptions, TaskGroup
# ---
# See: ../tutorial/05_Python并发编程.md#553-create_task-vs-gather-vs-taskgroup
# Interview hooks:
#   - create_task 和直接 await 一个协程对象的区别？
#   - gather(return_exceptions=True) 的语义？
#   - TaskGroup 相比 gather 的最大优势？异常* 语法是什么？
import asyncio


async def task(name, delay, fail=False):
    await asyncio.sleep(delay)
    if fail:
        raise ValueError(f"{name} 失败!")
    return f"{name} 完成"


async def main():
    # ========== create_task：创建后台任务 ==========
    # 适用于：需要立即启动但稍后 await 的场景
    task1 = asyncio.create_task(task("A", 1))
    task2 = asyncio.create_task(task("B", 2))

    # 此时 A 和 B 已经并行运行了
    print("任务已启动，可以做其他事...")

    result1 = await task1  # 等待 A 完成
    result2 = await task2  # 等待 B 完成
    print(f"{result1}, {result2}\n")

    # ========== gather：批量等待（Python 3.7+）==========
    # 适用于：同时启动多个任务，等待全部完成
    results = await asyncio.gather(
        task("C", 1),
        task("D", 2),
        task("E", 1),
        return_exceptions=True,  # 捕获异常而不是抛出
    )
    print(f"Gather 结果: {results}\n")

    # ========== TaskGroup（Python 3.11+）==========
    # 适用于：结构化并发，一个失败全部取消
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(task("F", 1))
            tg.create_task(task("G", 2))
            tg.create_task(task("H", 1, fail=True))  # 会触发失败
    except* Exception as eg:
        print(f"TaskGroup 异常: {eg.exceptions}")


if __name__ == "__main__":
    asyncio.run(main())
    print("OK")
