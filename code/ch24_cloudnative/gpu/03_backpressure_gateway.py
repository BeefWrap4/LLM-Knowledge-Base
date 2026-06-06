# ---
# chapter: 24
# topic: 云原生部署与工程化
# section: 24.5.3 请求队列与背压处理
# difficulty: ⭐⭐⭐⭐
# tier: gpu
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
import time
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class ReqPriority(Enum):
    HIGH = 0      # 实时对话
    NORMAL = 1    # API 调用
    LOW = 2       # 批量任务


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
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # 三级优先级队列
        self._queues = {
            ReqPriority.HIGH: deque(),
            ReqPriority.NORMAL: deque(),
            ReqPriority.LOW: deque(),
        }

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
        # 1. 队列满检查
        total_queued = sum(len(q) for q in self._queues.values())
        if total_queued >= self.max_queue_size:
            self.stats["rejected"] += 1
            return {"error": "queue_full", "retry_after_ms": 5000}

        # 2. 入队
        self._queues[request.priority].append(request)
        self.stats["accepted"] += 1

        # 3. 等待处理
        try:
            async with asyncio.timeout(self.timeout):
                async with self._semaphore:
                    # 从队列取出（按优先级）
                    req = self._dequeue()
                    self.stats["completed"] += 1
                    # 这里转发到实际的推理后端
                    return await self._forward_to_backend(req)
        except asyncio.TimeoutError:
            self.stats["timeout"] += 1
            return {"error": "timeout", "message": "Request timed out"}

    def _dequeue(self) -> InferenceRequest:
        """按优先级出队：高优先 > 普通 > 低优先"""
        for priority in (ReqPriority.HIGH, ReqPriority.NORMAL, ReqPriority.LOW):
            q = self._queues[priority]
            if q:
                return q.popleft()
        raise RuntimeError("Queue unexpectedly empty")

    async def _forward_to_backend(self, request: InferenceRequest) -> dict:
        """转发到推理后端（简化示例）"""
        # 实际实现中会调用 gRPC / HTTP 客户端
        await asyncio.sleep(0.5)  # 模拟推理耗时
        return {"request_id": request.request_id, "result": "generated..."}

    def get_stats(self) -> dict:
        """获取网关统计"""
        self.stats["current_queue_depth"] = sum(
            len(q) for q in self._queues.values()
        )
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

    # 触发队列满拒绝
    for i in range(20):
        await gw.submit(InferenceRequest(
            request_id=f"overflow-{i}",
            priority=ReqPriority.LOW,
            payload={"prompt": "x"},
        ))

    print("\nFinal stats:", gw.get_stats())


if __name__ == "__main__":
    asyncio.run(demo())
    print("OK")
