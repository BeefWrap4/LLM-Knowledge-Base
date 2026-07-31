# ---
# chapter: 24
# topic: 云原生部署与工程化
# section: 24.2.5 代码示例：FastAPI + vLLM 模型服务容器化
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: fastapi, uvicorn, pydantic, vllm
# run: MODEL_PATH=/models/your-model TENSOR_PARALLEL_SIZE=1 python 01_fastapi_vllm_server.py
# expected_runtime: 60-180s (model load) + interactive
# expected_output: model load succeeds, then /health reports ready and chat returns JSON
# ---
# See: ../tutorial/24_云原生部署与工程化.md §24.2.5
# Interview hooks:
#   1. FastAPI lifespan 与 vLLM AsyncLLMEngine 的生命周期如何配合？
#   2. 为什么用 asyncio.Semaphore 控制并发，而不是线程池？
#   3. Prefix Cache (enable_prefix_caching=True) 在大模型推理中能节省多少成本？
"""
FastAPI + vLLM 教学服务骨架。

这不是可直接暴露到公网的“生产级服务”：鉴权、限流、审计、模型原生 chat
template、可观测性与终止时的请求排空仍需按部署环境补齐。真实模式若 vLLM
或模型加载失败会立即退出，不会静默切换到伪推理。
"""

import os
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import skip_if_mock

if skip_if_mock("vLLM, a compatible GPU, local model weights, and a free HTTP port"):
    raise SystemExit(0)

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# 配置结构化日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("llm-server")


# ====== 配置 ======
class ServerConfig:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "").strip()
    TENSOR_PARALLEL_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    MAX_MODEL_LEN: int = int(os.getenv("MAX_MODEL_LEN", "32768"))
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "64"))
    PORT: int = int(os.getenv("PORT", "8000"))


config = ServerConfig()


# ====== 请求/响应模型 ======
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-72b"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False


# ====== 信号量控制并发 ======
semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)


# ====== 应用生命周期 ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭的生命周期管理"""
    if not config.MODEL_PATH:
        raise RuntimeError(
            "MODEL_PATH 未设置。请指向已获授权且与当前 vLLM 版本兼容的本地模型或 Hub 模型。"
        )

    logger.info(f"Loading model from {config.MODEL_PATH}...")
    t0 = time.time()

    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.sampling_params import SamplingParams

        engine_args = AsyncEngineArgs(
            model=config.MODEL_PATH,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            max_model_len=config.MAX_MODEL_LEN,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,
        )
        app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)
        app.state.sampling_params_class = SamplingParams
        app.state.engine_kind = "vllm"
        logger.info(f"Model loaded in {time.time() - t0:.1f}s. Ready to serve.")
    except Exception as e:
        logger.exception("vLLM/model initialization failed; refusing to start")
        raise RuntimeError(f"vLLM/model initialization failed: {e}") from e

    yield
    # 清理资源
    if hasattr(app.state, "engine"):
        del app.state.engine
    logger.info("Server shutting down.")


# ====== FastAPI 应用 ======
app = FastAPI(title="LLM Inference Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== 端点 ======
@app.get("/health")
async def health():
    """K8s Readiness / Liveness Probe 端点"""
    if not hasattr(app.state, "engine") or getattr(app.state, "engine_kind", None) != "vllm":
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "model_loaded": False, "engine": None},
        )
    return {"status": "ready", "model_loaded": True, "engine": "vllm"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容的 Chat Completions 端点"""
    if request.stream:
        return StreamingResponse(
            _stream_generate(request),
            media_type="text/event-stream",
        )

    async with semaphore:
        try:
            result = await _generate_completion(request)
            return JSONResponse(content=result)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ====== 推理核心 ======
async def _generate_completion(request: ChatCompletionRequest) -> dict:
    """同步（一次性）生成"""
    prompt = _build_prompt(request.messages)
    sampling_params = app.state.sampling_params_class(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
    )

    request_id = uuid.uuid4().hex
    final_output = None
    async for result in app.state.engine.generate(prompt, sampling_params, request_id):
        final_output = result

    if final_output is None:
        raise HTTPException(status_code=500, detail="Generation failed")

    completion_text = final_output.outputs[0].text
    return {
        "id": request_id,
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": len(final_output.outputs[0].token_ids),
            "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids),
        },
    }


async def _stream_generate(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """流式 SSE（Server-Sent Events）生成"""
    import json

    async with semaphore:
        try:
            prompt = _build_prompt(request.messages)
            sampling_params = app.state.sampling_params_class(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            )

            request_id = uuid.uuid4().hex
            previous_text = ""
            async for result in app.state.engine.generate(prompt, sampling_params, request_id):
                text = result.outputs[0].text
                delta = text[len(previous_text) :] if text.startswith(previous_text) else text
                previous_text = text
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


def _build_prompt(messages: list[ChatMessage]) -> str:
    """将消息列表构建为模型 Prompt"""
    return (
        "".join(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n" for msg in messages)
        + "<|im_start|>assistant\n"
    )


# ====== 入口 ======
if __name__ == "__main__":
    print("Starting LLM Inference Server (use curl http://localhost:8000/health to test)")
    print("Open: http://localhost:8000/docs for interactive Swagger UI")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        log_level="info",
        limit_concurrency=config.MAX_CONCURRENT_REQUESTS,
    )
