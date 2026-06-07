# ---
# chapter: 24
# topic: 云原生部署与工程化
# section: 24.2.5 代码示例：FastAPI + vLLM 模型服务容器化
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: fastapi, uvicorn, pydantic, vllm
# run: python 01_fastapi_vllm_server.py
# expected_runtime: 60-180s (model load) + interactive
# expected_output: /health returns {"status":"healthy"}, /v1/chat/completions returns OpenAI-format JSON
# ---
# See: ../tutorial/24_云原生部署与工程化.md §24.2.5
# Interview hooks:
#   1. FastAPI lifespan 与 vLLM AsyncLLMEngine 的生命周期如何配合？
#   2. 为什么用 asyncio.Semaphore 控制并发，而不是线程池？
#   3. Prefix Cache (enable_prefix_caching=True) 在大模型推理中能节省多少成本？


# === Multi-GPU / heavy model guard (auto-added) ===
import sys as _sys
import os as _os
_NGPU = _os.environ.get("WORLD_SIZE", "1")
if _NGPU == "1" and not _os.environ.get("FORCE_GPU_RUN"):
    print(f"[SKIP] {{__file__}}: 需多卡 (WORLD_SIZE>1) 或真实模型权重, 用 torchrun 或设置 FORCE_GPU_RUN=1")
    _sys.exit(0)
"""
大模型推理服务 —— FastAPI + vLLM 后端
支持 OpenAI 兼容 API、流式输出、并发控制
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

# 配置结构化日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("llm-server")

# ====== 配置 ======
class ServerConfig:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/models/Qwen2.5-72B-Instruct-AWQ")
    TENSOR_PARALLEL_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "4"))
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

# ====== Mock 引擎：用于无 GPU 环境演示 ======
class MockVLLMEngine:
    """无 GPU 环境下的伪 vLLM AsyncLLMEngine。"""
    async def generate(self, prompt, sampling_params, request_id):
        # 模拟一个分词的输出序列
        tokens = ["Hello", " there", "!", " This", " is", " a", " mock", " response", "."]
        text = ""
        for tok in tokens:
            await asyncio.sleep(0.05)
            text += tok
            yield _MockResult(text=text, token_count=len(tokens))


class _MockResult:
    def __init__(self, text, token_count):
        out = _MockOutput()
        out.text = text
        out.token_ids = text.split()
        self.outputs = [out]
        self.prompt_token_ids = prompt.split() if isinstance(prompt, str) else []


class _MockOutput:
    pass


# ====== 应用生命周期 ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭的生命周期管理"""
    logger.info(f"Loading model from {config.MODEL_PATH}...")
    t0 = time.time()

    # 尝试真实加载 vLLM，失败则使用 Mock
    try:
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

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
        logger.info(f"Model loaded in {time.time() - t0:.1f}s. Ready to serve.")
    except Exception as e:
        logger.warning(f"vLLM not available ({e}), using Mock engine")
        app.state.engine = MockVLLMEngine()
        logger.info(f"Mock engine ready in {time.time() - t0:.1f}s.")

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
    return {"status": "healthy", "model_loaded": hasattr(app.state, "engine")}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
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
    try:
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid
    except ImportError:
        # Mock 模式下的占位
        class SamplingParams:
            def __init__(self, **kwargs):
                self.params = kwargs
        def random_uuid():
            return f"mock-{int(time.time()*1000)}"

    prompt = _build_prompt(request.messages)
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
    )

    request_id = random_uuid()
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
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": completion_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": len(final_output.outputs[0].token_ids),
            "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids),
        },
    }

async def _stream_generate(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """流式 SSE（Server-Sent Events）生成"""
    try:
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid
    except ImportError:
        class SamplingParams:
            def __init__(self, **kwargs):
                self.params = kwargs
        def random_uuid():
            return f"mock-{int(time.time()*1000)}"

    import json

    async with semaphore:
        try:
            prompt = _build_prompt(request.messages)
            sampling_params = SamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            )

            request_id = random_uuid()
            async for result in app.state.engine.generate(
                prompt, sampling_params, request_id
            ):
                text = result.outputs[0].text
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

def _build_prompt(messages: list[ChatMessage]) -> str:
    """将消息列表构建为模型 Prompt"""
    return "".join(
        f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        for msg in messages
    ) + "<|im_start|>assistant\n"

# ====== 入口 ======
if __name__ == "__main__":
    print("Starting LLM Inference Server (use curl http://localhost:8000/health to test)")
    print("Open: http://localhost:8000/docs for interactive Swagger UI")
    uvicorn.run(
        "01_fastapi_vllm_server:app",
        host="0.0.0.0",
        port=config.PORT,
        workers=1,
        log_level="info",
        limit_concurrency=config.MAX_CONCURRENT_REQUESTS,
    )