# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.7.4 分布式处理框架 - Ray 实战
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


import os


class LocalTextProcessor:
    """无 Ray 环境下复用同一批处理语义。"""

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self.processed_count = 0

    def process_batch(self, texts: list[str]) -> list[dict]:
        results = []
        for text in texts:
            self.processed_count += 1
            results.append(
                {
                    "text": text,
                    "length": len(text),
                    "word_count": len(text.split()),
                    "lang": self.lang,
                }
            )
        return results

    def get_count(self) -> int:
        return self.processed_count


def build_batches() -> list[list[str]]:
    return [[f"这是第{i}批的第{j}条数据，用于测试 Ray 分布式处理。" for j in range(100)] for i in range(8)]


def run_local_demo() -> tuple[int, list[int]]:
    num_workers = 4
    processors = [LocalTextProcessor() for _ in range(num_workers)]
    all_results = [
        processors[i % num_workers].process_batch(batch) for i, batch in enumerate(build_batches())
    ]
    return sum(len(batch) for batch in all_results), [processor.get_count() for processor in processors]


def run_ray_demo() -> tuple[int, list[int]]:
    import ray  # type: ignore

    @ray.remote
    class RayTextProcessor:
        """有状态 Actor：每个 worker 独立维护处理计数。"""

        def __init__(self, lang: str = "zh"):
            self.lang = lang
            self.processed_count = 0

        def process_batch(self, texts: list[str]) -> list[dict]:
            results = []
            for text in texts:
                self.processed_count += 1
                results.append(
                    {
                        "text": text,
                        "length": len(text),
                        "word_count": len(text.split()),
                        "lang": self.lang,
                    }
                )
            return results

        def get_count(self) -> int:
            return self.processed_count

    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False)
    try:
        num_workers = 4
        processors = [RayTextProcessor.remote() for _ in range(num_workers)]
        futures = [
            processors[i % num_workers].process_batch.remote(batch)
            for i, batch in enumerate(build_batches())
        ]
        all_results = ray.get(futures)
        counts = ray.get([processor.get_count.remote() for processor in processors])
        return sum(len(batch) for batch in all_results), counts
    finally:
        ray.shutdown()


def main() -> None:
    use_local = os.environ.get("LLM_MOCK") == "1"
    if not use_local:
        try:
            import ray  # type: ignore  # noqa: F401
        except ImportError:
            use_local = True

    if use_local:
        print("[本地模式] Ray 未启用；以相同分片逻辑验证处理流程")
        total_processed, counts = run_local_demo()
    else:
        total_processed, counts = run_ray_demo()

    print(f"总共处理了 {total_processed} 条数据")
    print(f"各 Worker 处理量: {counts}")
    print("OK")


if __name__ == "__main__":
    main()
