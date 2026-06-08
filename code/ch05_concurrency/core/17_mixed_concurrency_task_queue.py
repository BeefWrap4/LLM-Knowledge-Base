# ---
# chapter: 05
# topic: Python并发编程
# section: 5.6.3 混合并发模式实战 + 生产级异步任务队列
# difficulty: ⭐⭐⭐
# tier: core
# deps: [aiohttp]
# run: python 17_mixed_concurrency_task_queue.py
# expected_runtime: ~3s
# expected_output: mixed-concurrency stub + TaskQueue demo with 20 tasks
# ---
# See: ../tutorial/05_Python并发编程.md#563-混合并发模式实战
# Interview hooks:
#   - 混合并发中 IO 和 CPU 任务的边界划分原则？
#   - 自建 TaskQueue 和直接 asyncio.gather 的区别？
#   - asyncio.Queue + Semaphore 实现"有限并发"的常见模式？
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum


# ========== 混合并发：IO 协程 + CPU 进程池 ==========
def cpu_intensive(data: dict) -> dict:
    """CPU 密集型：数据处理"""
    # 模拟复杂计算
    result = {k: v**2 for k, v in data.items() if isinstance(v, (int, float))}
    time.sleep(0.05)  # 模拟计算时间
    return result


async def fetch_data(session, url: str) -> dict:
    """IO 密集型：网络请求（stub）"""
    # 真实环境使用 aiohttp，这里仅演示流程
    await asyncio.sleep(0.01)
    return {"id": url, "value": 1}


async def process_urls(urls: list) -> list:
    """
    混合并发模式：
    - IO 部分用 asyncio（协程处理 HTTP 请求）
    - CPU 部分用 ProcessPoolExecutor（多进程处理数据）
    """
    # Step 1: 协程并发获取数据（IO 密集型）
    async with aiohttp_helper() as session:
        fetch_tasks = [fetch_data(session, url) for url in urls]
        raw_data_list = await asyncio.gather(*fetch_tasks)

    # Step 2: 进程池并行处理数据（CPU 密集型）
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=2) as pool:
        process_tasks = [loop.run_in_executor(pool, cpu_intensive, data) for data in raw_data_list]
        processed_data = await asyncio.gather(*process_tasks)

    return processed_data


# 不依赖 aiohttp 的兼容上下文管理器
class _DummySession:
    async def get(self, url):
        class _Resp:
            async def json(self_inner):
                return {"id": url, "value": 1}

        return _Resp()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


# (原 @asyncio.coroutine 已删除 — Python 3.11+ 完全移除, 用 async def 替代)


def aiohttp_helper():
    """如果安装了 aiohttp 则使用真 session，否则用 stub。"""
    try:
        import aiohttp  # noqa: F401
        import aiohttp as _aio

        return _aio.ClientSession()
    except Exception:
        return _DummySession()


# ========== 生产级并发模式：后台任务队列 ==========
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    coro: "asyncio.coroutines"  # 协程对象
    status: TaskStatus = TaskStatus.PENDING
    result: object = None


class TaskQueue:
    """生产级异步任务队列"""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.queue = asyncio.Queue()
        self.tasks = {}
        self.semaphore = asyncio.Semaphore(max_workers)

    async def submit(self, task_id: str, coro) -> Task:
        """提交任务到队列"""
        task = Task(id=task_id, coro=coro)
        self.tasks[task_id] = task
        await self.queue.put(task)
        return task

    async def _worker(self):
        """工作协程：从队列取任务执行"""
        while True:
            task = await self.queue.get()
            async with self.semaphore:
                task.status = TaskStatus.RUNNING
                try:
                    task.result = await task.coro
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.result = e
                    task.status = TaskStatus.FAILED
                finally:
                    self.queue.task_done()

    async def start(self):
        """启动工作协程"""
        workers = [asyncio.create_task(self._worker()) for _ in range(self.max_workers)]
        return workers

    async def wait_all(self):
        """等待所有任务完成"""
        await self.queue.join()


# 使用示例
async def demo_task_queue():
    async def my_task(name, delay):
        await asyncio.sleep(delay)
        return f"{name} done"

    queue = TaskQueue(max_workers=5)
    workers = await queue.start()

    # 提交 20 个任务
    for i in range(20):
        await queue.submit(f"task-{i}", my_task(f"Task-{i}", 0.01))

    await queue.wait_all()

    completed = sum(1 for t in queue.tasks.values() if t.status == TaskStatus.COMPLETED)
    print(f"完成 {completed}/{len(queue.tasks)} 个任务")

    # 清理工作协程
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


async def main():
    # 混合并发演示（使用 stub 网络，跨环境可跑）
    demo_urls = [f"url-{i}" for i in range(5)]
    processed = await process_urls(demo_urls)
    print(f"混合并发处理了 {len(processed)} 个任务")

    # 任务队列演示
    await demo_task_queue()


if __name__ == "__main__":
    asyncio.run(main())
