# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.6.4 分布式处理框架 - Ray 实战
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: ray
# run: python 13_ray_distributed_processing.py
# expected_runtime: 30-60s (ray 启动)
# expected_output: 总处理条数 + 各 Worker 处理量列表
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. Ray 的 Actor 模型与 Spark 的 RDD 模型在数据处理上有何本质差异？
#   2. 为什么 Agent / LLM 场景下 Ray 增长快于 Spark？Actor 模式优势在哪？
#   3. 分布式数据处理中如何处理倾斜 (skew) 问题？Round-robin 分片够用吗？

import sys
from typing import List, Dict

# 提供无 ray 环境下的优雅降级
try:
    import ray  # type: ignore
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("⚠️  ray 未安装，将以本地单进程模式演示（用于展示逻辑）")


if RAY_AVAILABLE:
    # 初始化 Ray（单机或集群）
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False)

    @ray.remote
    class TextProcessor:
        """分布式文本处理器"""

        def __init__(self, lang: str = "en"):
            self.lang = lang
            self.processed_count = 0

        def process_batch(self, texts: List[str]) -> List[Dict]:
            """处理一批文本"""
            results = []
            for text in texts:
                self.processed_count += 1
                results.append({
                    "text": text,
                    "length": len(text),
                    "word_count": len(text.split()),
                    "lang": self.lang,
                })
            return results

        def get_count(self) -> int:
            return self.processed_count

    # 创建多个分布式 Worker
    num_workers = 4
    processors = [TextProcessor.remote(lang="en") for _ in range(num_workers)]

    # 模拟数据批次
    batches = [
        [f"这是第{i}批的第{j}条数据，用于测试Ray分布式处理。" for j in range(100)]
        for i in range(8)
    ]

    # 分布式处理（Round-robin 分配批次）
    futures = []
    for i, batch in enumerate(batches):
        worker_idx = i % num_workers
        future = processors[worker_idx].process_batch.remote(batch)
        futures.append(future)

    # 收集结果
    all_results = ray.get(futures)
    total_processed = sum(len(batch) for batch in all_results)
    print(f"总共处理了 {total_processed} 条数据")

    # 查询每个 worker 的处理量
    counts = ray.get([p.get_count.remote() for p in processors])
    print(f"各 Worker 处理量: {counts}")

    ray.shutdown()
else:
    # 本地降级演示
    class TextProcessor:
        def __init__(self, lang="en"):
            self.lang = lang
            self.processed_count = 0

        def process_batch(self, texts):
            results = []
            for t in texts:
                self.processed_count += 1
                results.append({"text": t, "length": len(t),
                                "word_count": len(t.split()), "lang": self.lang})
            return results

        def get_count(self):
            return self.processed_count

    num_workers = 4
    processors = [TextProcessor(lang="en") for _ in range(num_workers)]
    batches = [[f"文本{j}" for j in range(100)] for _ in range(8)]
    all_results = []
    for i, batch in enumerate(batches):
        all_results.append(processors[i % num_workers].process_batch(batch))
    total_processed = sum(len(b) for b in all_results)
    counts = [p.get_count() for p in processors]
    print(f"[本地模式] 总共处理了 {total_processed} 条数据")
    print(f"[本地模式] 各 Worker 处理量: {counts}")

print("OK")
