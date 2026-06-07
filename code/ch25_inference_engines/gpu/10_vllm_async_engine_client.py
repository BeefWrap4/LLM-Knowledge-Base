# ---
# chapter: 25
# topic: vLLM Async Engine Client (Mock)
# section: 25.2.1 / 25.6
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: none (mock; real usage: vllm >= 0.4)
# run: python 10_vllm_async_engine_client.py
# expected_runtime: <1s
# expected_output: 模拟 AsyncLLMEngine 的流式输出 + 多请求并发
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.2.1, §25.6
# Interview hooks:
#   1. vLLM 0.x → 0.4+ 的 API 变化？(答: LLMEngine → AsyncLLMEngine, 同步→async stream)
#   2. SamplingParams 的关键字段？(答: temperature, top_p, top_k, max_tokens, stop)
#   3. vLLM 如何做 OpenAI 兼容 server？(答: --served-model-name + FastAPI 包装)

"""Mock the vLLM AsyncLLMEngine interface.

The real class is `vllm.AsyncLLMEngine`; this file shows the *contract*
you would code against, so the interview discussion is concrete.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 64
    stop: list[str] = field(default_factory=list)


@dataclass
class RequestOutput:
    request_id: str
    text: str = ""
    finished: bool = False
    usage: dict = field(default_factory=dict)


class AsyncLLMEngine:
    """Mimics `vllm.AsyncLLMEngine.generate` API surface."""

    def __init__(self, model: str, max_num_seqs: int = 64) -> None:
        self.model = model
        self.max_num_seqs = max_num_seqs
        self._inflight = 0
        print(f"[engine] loaded {model}  max_num_seqs={max_num_seqs}")

    async def generate(
        self, prompt: str, sampling: SamplingParams, request_id: str
    ) -> AsyncIterator[RequestOutput]:
        if self._inflight >= self.max_num_seqs:
            yield RequestOutput(request_id=request_id, text="<rejected: backpressure>")
            return
        self._inflight += 1
        try:
            out = RequestOutput(request_id=request_id)
            # mock token stream
            tokens = [f"tok{i}-" for i in range(sampling.max_tokens)]
            cur = ""
            for t in tokens:
                cur += t
                out.text = cur
                out.usage = {"prompt_tokens": len(prompt.split()),
                             "completion_tokens": len(cur.split("-"))}
                # simulate decode step
                await asyncio.sleep(0)
                yield out
            out.finished = True
            yield out
        finally:
            self._inflight -= 1


async def main() -> None:
    engine = AsyncLLMEngine(model="llama-3-8b-instruct", max_num_seqs=4)
    sp = SamplingParams(temperature=0.7, max_tokens=8)

    async def one_call(rid: str, prompt: str) -> None:
        async for out in engine.generate(prompt, sp, request_id=rid):
            if out.finished:
                print(f"[{rid}] DONE -> {out.text[:60]}...  usage={out.usage}")

    # fire 3 requests concurrently
    await asyncio.gather(
        one_call("r1", "What is PagedAttention?"),
        one_call("r2", "Explain speculative decoding."),
        one_call("r3", "Why is decode memory-bound?"),
    )


if __name__ == "__main__":
    asyncio.run(main())
