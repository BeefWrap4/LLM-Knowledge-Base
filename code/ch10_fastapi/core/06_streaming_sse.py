# ---
# chapter: 10
# topic: FastAPI 与后端服务
# topic_id: fastapi.streaming_sse
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: fastapi
# run: python 06_streaming_sse.py
# expected_runtime: ~2s
# expected_output: 全部 SSE chunk 按顺序产出
# ---
# See: ../../../10_FastAPI与后端服务.md
# Interview hooks:
#   1. SSE 与 WebSocket 在协议层和适用场景上的核心差异？
#   2. StreamingResponse 的 media_type="text/event-stream" 起到什么作用？
#   3. 在 yield 异步生成器中如何优雅处理客户端断连 (GeneratorExit)？
import asyncio
import json
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Streaming Demo")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8_000)


async def generate_tokens_stream(
    prompt: str,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
):
    """模拟 LLM 流式生成"""
    tokens = [
        "Fast",
        "API",
        "是",
        "一个",
        "现代",
        "、",
        "高性能",
        "的",
        "Python",
        "Web",
        "框架",
        "，",
        "特别适合",
        "构建",
        "LLM",
        "服务",
        "。",
    ]
    full_response = ""
    for token in tokens:
        if is_disconnected is not None and await is_disconnected():
            return
        await asyncio.sleep(0.05)  # 模拟推理延迟（示例加速）
        full_response += token
        chunk = {"token": token, "choices": [{"delta": {"content": token}}]}
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    # 发送结束标记
    yield f"data: {json.dumps({'done': True, 'full_text': full_response})}\n\n"


@app.post("/chat/stream", summary="流式对话（SSE over fetch）")
async def chat_stream(payload: ChatRequest, request: Request):
    """
    POST body 避免把 prompt 放入 URL；浏览器可用 fetch 读取响应流。
    生产版本还应在此依赖认证、限流，并将断连取消传播到模型服务。
    """
    return StreamingResponse(
        generate_tokens_stream(payload.message, request.is_disconnected),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _drain_stream():
    """在脚本中直接消费异步生成器，模拟客户端读取."""
    chunks = []
    async for chunk in generate_tokens_stream("hi"):
        chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    # 1. 路由注册检查
    paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/chat/stream" in paths
    stream_route = next(r for r in app.routes if getattr(r, "path", None) == "/chat/stream")
    assert stream_route.methods == {"POST"}
    print(f"路由已注册: {paths}")

    # 2. 跑通异步生成器
    chunks = asyncio.run(_drain_stream())
    assert all(c.startswith("data: ") for c in chunks), "SSE 帧必须以 'data: ' 开头"
    assert chunks[-1].endswith("\n\n"), "SSE 帧必须以双换行结束"

    # 3. 解析 payload，验证 done 标记与完整文本
    payloads = [json.loads(c.removeprefix("data: ").strip()) for c in chunks]
    last = payloads[-1]
    assert last.get("done") is True
    assert last["full_text"].endswith("LLM服务。"), (
        f"full_text 应以 'LLM服务。' 结尾, 实际: {last['full_text']!r}"
    )
    print(f"累计产出 {len(chunks)} 个 SSE chunk，完整文本: {last['full_text']}")
    print("OK")
