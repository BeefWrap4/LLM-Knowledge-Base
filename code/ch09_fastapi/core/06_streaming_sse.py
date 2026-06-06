# ---
# chapter: 9
# topic: Web开发与FastAPI
# section: 9.3.5 流式响应（SSE）— LLM 场景核心
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: fastapi
# run: python 06_streaming_sse.py
# expected_runtime: ~2s
# expected_output: 全部 SSE chunk 按顺序产出
# ---
# See: ../tutorial/09_Web开发与FastAPI.md (lines 366-398)
# Interview hooks:
#   1. SSE 与 WebSocket 在协议层和适用场景上的核心差异？
#   2. StreamingResponse 的 media_type="text/event-stream" 起到什么作用？
#   3. 在 yield 异步生成器中如何优雅处理客户端断连 (GeneratorExit)？
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI(title="LLM Streaming Demo")

async def generate_tokens_stream(prompt: str):
    """模拟 LLM 流式生成"""
    tokens = ["Fast", "API", "是", "一个", "现代", "、", "高性能", "的",
              "Python", "Web", "框架", "，", "特别适合", "构建", "LLM", "服务", "。"]
    full_response = ""
    for token in tokens:
        await asyncio.sleep(0.05)  # 模拟推理延迟（示例加速）
        full_response += token
        chunk = {
            "token": token,
            "choices": [{"delta": {"content": token}}]
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    # 发送结束标记
    yield f"data: {json.dumps({'done': True, 'full_text': full_response})}\n\n"

@app.get("/chat/stream", summary="流式对话（SSE）")
async def chat_stream(message: str = Query(min_length=1)):
    """
    SSE 流式响应 - 大模型 API 的标准输出方式
    与 WebSocket 的区别：SSE 是单向（服务端→客户端），基于 HTTP
    """
    return StreamingResponse(
        generate_tokens_stream(message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
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
    print(f"路由已注册: {paths}")

    # 2. 跑通异步生成器
    chunks = asyncio.run(_drain_stream())
    assert all(c.startswith("data: ") for c in chunks), "SSE 帧必须以 'data: ' 开头"
    assert chunks[-1].endswith("\n\n"), "SSE 帧必须以双换行结束"

    # 3. 解析 payload，验证 done 标记与完整文本
    payloads = [json.loads(c.removeprefix("data: ").strip()) for c in chunks]
    last = payloads[-1]
    assert last.get("done") is True
    assert last["full_text"].endswith("LLM 服务。")
    print(f"累计产出 {len(chunks)} 个 SSE chunk，完整文本: {last['full_text']}")
    print("OK")
