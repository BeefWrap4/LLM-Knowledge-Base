# ---
# chapter: 24
# topic: 云原生部署与工程化
# section: 24.5.3 请求队列与背压处理
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: (stdlib only)
# run: python 03_backpressure_gateway.py
# expected_runtime: ~5s
# expected_output: Stats printed for 4 priority requests (HIGH/NORMAL/LOW) with backpressure metrics
# ---
# See: ../tutorial/24_云原生部署与工程化.md §24.5.3
# Interview hooks:
#   1. 三级优先级队列如何避免"低优先级饿死"？
#   2. asyncio.Semaphore 与 max_queue_size 的两层背压设计意图？
#   3. 真实生产中 queue_full 应返回 429 还是 503？两者语义差异？
"""
模型网关中的请求队列与背压处理
"""

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from enum import Enum


class ReqPriority(Enum):
    HIGH = 0  # 实时对话
    NORMAL = 1  # API 调用
    LOW = 2  # 批量任务


@dataclass
class InferenceRequest:
    request_id: str
    priority: ReqPriority
    payload: dict
    enqueue_time: float = field(default_factory=time.time)


class BackpressureGateway:
    """带背压控制的推理请求网关"""

    def __init__(
        self,
        max_queue_size: int = 100,
        max_concurrent: int = 32,
        timeout_seconds: float = 30.0,
    ):
        self.max_queue_size = max_queue_size
        self.max_concurrent = max_concurrent
        self.timeout = timeout_seconds
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._sequence = itertools.count()
        self._workers: list[asyncio.Task] = []

        # 统计指标
        self.stats = {
            "accepted": 0,
            "rejected": 0,
            "timeout": 0,
            "completed": 0,
            "current_queue_depth": 0,
        }

    async def submit(self, request: InferenceRequest) -> dict:
        """提交推理请求（带背压）"""
        self._ensure_workers()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = (request.priority.value, next(self._sequence), request, future)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self.stats["rejected"] += 1
            return {"error": "queue_full", "retry_after_ms": 5000}
        self.stats["accepted"] += 1

        try:
            # wait_for 支持 Python 3.10；shield 避免超时取消 worker 正在完成的 Future。
            return await asyncio.wait_for(asyncio.shield(future), timeout=self.timeout)
        except asyncio.TimeoutError:
            self.stats["timeout"] += 1
            future.cancel()
            return {"error": "timeout", "message": "Request timed out"}

    def _ensure_workers(self) -> None:
        if not self._workers:
            self._workers = [
                asyncio.create_task(self._worker(), name=f"gateway-worker-{index}")
                for index in range(self.max_concurrent)
            ]

    async def _worker(self) -> None:
        while True:
            _, _, request, future = await self._queue.get()
            try:
                if not future.cancelled():
                    result = await self._forward_to_backend(request)
                    self.stats["completed"] += 1
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def close(self) -> None:
        """完成排队任务并停止 worker；生产服务应在 shutdown hook 中调用。"""
        await self._queue.join()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def _forward_to_backend(self, request: InferenceRequest) -> dict:
        """转发到推理后端（简化示例）"""
        # 实际实现中会调用 gRPC / HTTP 客户端
        await asyncio.sleep(0.5)  # 模拟推理耗时
        return {"request_id": request.request_id, "result": "generated..."}

    def get_stats(self) -> dict:
        """获取网关统计"""
        self.stats["current_queue_depth"] = self._queue.qsize()
        return self.stats


# ====== 演示用法 ======
async def demo():
    """运行一个最小可执行演示，验证队列/信号量/背压行为。"""
    gw = BackpressureGateway(max_queue_size=10, max_concurrent=2, timeout_seconds=2.0)

    # 提交不同优先级请求
    reqs = [
        InferenceRequest(request_id="r1", priority=ReqPriority.HIGH, payload={"prompt": "Hi"}),
        InferenceRequest(request_id="r2", priority=ReqPriority.NORMAL, payload={"prompt": "Hello"}),
        InferenceRequest(request_id="r3", priority=ReqPriority.LOW, payload={"prompt": "Batch task"}),
        InferenceRequest(request_id="r4", priority=ReqPriority.HIGH, payload={"prompt": "Realtime"}),
    ]

    # 并发提交
    results = await asyncio.gather(*(gw.submit(r) for r in reqs), return_exceptions=True)
    for r, req in zip(results, reqs):
        print(f"[{req.priority.name}] {req.request_id}: {r}")

    assert all(
        result["request_id"] == request.request_id
        for result, request in zip(results, reqs, strict=True)
    )
    await gw.close()
    print("\nFinal stats:", gw.get_stats())
    print("OK")


if __name__ == "__main__":
    asyncio.run(demo())
