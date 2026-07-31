# ---
# chapter: 25
# topic: TTFT / TPOT SLO Monitor (真实 prometheus_client)
# section: 25.6
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: prometheus-client
# run: python 11_slo_ttft_tpot_monitor.py
# expected_runtime: 5-30s (10 mock requests + http server)
# expected_output: 启动 :9090 Prometheus 端点, 10 个 mock 请求的 TTFT/TPOT 记录
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.6
# Interview hooks:
#   1. 什么是 TTFT 和 TPOT？分别由什么决定？
#   2. 如何定义 LLM 服务 SLO？p50/p99 各自代表什么？
#   3. SLO 不达标时如何定位？(答: prefill/decode 拆分分析、batching 延迟、KV 压力)
"""SLO 监控: TTFT (Time To First Token) + TPOT (Time Per Output Token).

真实暴露 Prometheus metrics (HTTP :9090/metrics), 配合 Grafana 可视化.
本文件模拟 10 个 LLM 请求, 记录 SLO 指标. 生产环境替换 simulate_request()
为真实 vLLM 调用即可.
"""

import random
import sys
import time
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import skip_if_mock

try:
    from prometheus_client import Counter, Histogram, start_http_server

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


def setup_metrics():
    """定义 Prometheus metrics."""
    if not HAS_PROMETHEUS:
        return None, None, None
    return (
        Histogram("llm_ttft_seconds", "Time to first token", buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0)),
        Histogram("llm_tpot_seconds", "Time per output token", buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5)),
        Counter("llm_requests_total", "Total LLM requests"),
    )


def simulate_request(req_id: int, ttft_hist, tpot_hist, req_counter) -> dict:
    """模拟单次 LLM 请求, 记录 SLO metrics.

    生产替换: 真实 vllm/tgi 调用即可, 仍用 Histogram.observe().
    """
    if not HAS_PROMETHEUS:
        return {}

    # 模拟 TTFT (50-300ms)
    ttft = random.uniform(0.05, 0.3)
    time.sleep(ttft * 0.01)  # 加速模拟 (实际会真 sleep)
    ttft_hist.observe(ttft)

    # 模拟 TPOT (10-100ms per token)
    num_tokens = random.randint(50, 200)
    total_tpot = 0
    for _ in range(num_tokens):
        tpot = random.uniform(0.01, 0.1)
        total_tpot += tpot
        tpot_hist.observe(tpot)
    avg_tpot_ms = total_tpot / num_tokens * 1000

    req_counter.inc()
    print(
        f"  [{req_id:2d}] TTFT={ttft * 1000:5.0f}ms | {num_tokens:3d} tokens | avg TPOT={avg_tpot_ms:5.1f}ms"
    )
    return {"ttft_ms": ttft * 1000, "tpot_ms": avg_tpot_ms, "tokens": num_tokens}


def main():
    if skip_if_mock("a free localhost metrics port and the optional prometheus-client dependency"):
        return
    if not HAS_PROMETHEUS:
        raise_with_help(
            "prometheus_client 未装",
            "运行 `pip install prometheus-client`.",
        )

    ttft_hist, tpot_hist, req_counter = setup_metrics()

    # 启动 Prometheus HTTP server
    port = 9090
    start_http_server(port)
    print(f"✅ Prometheus metrics server: http://localhost:{port}/metrics")
    print()
    print("=== 模拟 10 个 LLM 请求 ===")

    results = []
    for i in range(1, 11):
        r = simulate_request(i, ttft_hist, tpot_hist, req_counter)
        if r:
            results.append(r)

    # 汇总
    if results:
        avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
        avg_tpot = sum(r["tpot_ms"] for r in results) / len(results)
        print()
        print("=== SLO 汇总 (10 请求) ===")
        print(f"  avg TTFT: {avg_ttft:.0f}ms (SLO < 200ms) {'✅' if avg_ttft < 200 else '❌'}")
        print(f"  avg TPOT: {avg_tpot:.1f}ms (SLO < 50ms) {'✅' if avg_tpot < 50 else '❌'}")
        print()
        print("✅ Metrics 暴露在 :9090/metrics (Grafana 可抓取)")
        print()
        print("生产部署: 把 simulate_request() 替换为:")
        print("  - vllm.AsyncLLMEngine.generate() + timing")
        print("  - 或 tgi/trtllm-serve 的 OpenAI 协议 + 客户端计时")


if __name__ == "__main__":
    main()
