# ---
# chapter: 43
# topic: 云原生部署与模型网关
# topic_id: cloudnative.grpc_vllm_server
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: grpcio, grpcio-tools, vllm, protobuf
# run: MODEL_PATH=/models/your-model python 02_grpc_vllm_server.py
# expected_runtime: 60-180s (model load) + blocking
# expected_output: real vLLM engine loads, then gRPC server listens on :50051
# ---
# See: ../../../43_云原生部署与模型网关.md
# Interview hooks:
#   1. gRPC Server Streaming 与 SSE 的性能差异主要来自哪里？
#   2. 为什么大模型推理需要 keepalive 配置，keepalive_time_ms 该如何选？
#   3. Protobuf 与 JSON 在大模型消息体积上的差异如何量化？
"""
gRPC + vLLM 教学服务骨架。

先在本目录执行：
    python -m grpc_tools.protoc -I protos --python_out=. \
        --grpc_python_out=. protos/llm_inference.proto

真实模式缺依赖、生成代码、模型或 GPU 时会立即失败，不会以 mock 响应冒充服务可用。
示例使用未加密端口且未实现鉴权，只适合隔离的本地实验环境。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import skip_if_mock

if skip_if_mock("compiled protobuf modules, vLLM, a compatible GPU, model weights, and port 50051"):
    raise SystemExit(0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("grpc-llm-server")


def _load_runtime() -> tuple[Any, Any, Any, Any, Any]:
    """加载真实运行时；任何缺失都失败关闭。"""
    try:
        import grpc
    except ImportError as exc:
        raise RuntimeError("缺少 grpcio；请安装 GPU tier 依赖。") from exc

    try:
        import llm_inference_pb2 as pb2
        import llm_inference_pb2_grpc as pb2_grpc
    except ImportError as exc:
        raise RuntimeError(
            "缺少 protobuf 生成代码。请在本目录运行："
            "`python -m grpc_tools.protoc -I protos --python_out=. "
            "--grpc_python_out=. protos/llm_inference.proto`。"
        ) from exc

    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.sampling_params import SamplingParams
    except ImportError as exc:
        raise RuntimeError("缺少兼容版本的 vLLM；请安装并核对当前 vLLM 文档。") from exc

    return grpc, pb2, pb2_grpc, AsyncEngineArgs, (AsyncLLMEngine, SamplingParams)


def _build_prompt(messages: Any) -> str:
    """教学用 Qwen 风格模板；生产服务应改用目标 tokenizer 的 chat template。"""
    parts = [f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n" for msg in messages]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _sampling_params(request: Any, sampling_params_class: Any) -> Any:
    params = request.sampling_params
    return sampling_params_class(
        temperature=params.temperature if params.temperature > 0 else 0.7,
        max_tokens=params.max_tokens or 2048,
        top_p=params.top_p if params.top_p > 0 else 1.0,
        top_k=params.top_k or -1,
    )


def _make_servicer(
    *,
    grpc: Any,
    pb2: Any,
    pb2_grpc: Any,
    engine: Any,
    sampling_params_class: Any,
) -> Any:
    class LLMInferenceServicer(pb2_grpc.LLMInferenceServicer):
        async def Generate(self, request: Any, context: Any) -> Any:
            request_id = request.request_id or uuid.uuid4().hex
            prompt = _build_prompt(request.messages)
            final_output = None
            try:
                async for result in engine.generate(
                    prompt,
                    _sampling_params(request, sampling_params_class),
                    request_id,
                ):
                    final_output = result
            except Exception as exc:
                logger.exception("generation failed")
                await context.abort(grpc.StatusCode.INTERNAL, f"generation failed: {exc}")

            if final_output is None:
                await context.abort(grpc.StatusCode.INTERNAL, "generation returned no output")

            output = final_output.outputs[0]
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(output.token_ids)
            return pb2.GenerateResponse(
                request_id=request_id,
                completion=[
                    pb2.Completion(index=0, text=output.text, finish_reason="stop")
                ],
                usage=pb2.TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

        async def GenerateStream(self, request: Any, context: Any):
            request_id = request.request_id or uuid.uuid4().hex
            prompt = _build_prompt(request.messages)
            previous_text = ""
            try:
                async for result in engine.generate(
                    prompt,
                    _sampling_params(request, sampling_params_class),
                    request_id,
                ):
                    text = result.outputs[0].text
                    delta = text[len(previous_text) :] if text.startswith(previous_text) else text
                    previous_text = text
                    yield pb2.GenerateStreamResponse(
                        request_id=request_id,
                        token=pb2.TokenDelta(index=0, delta_text=delta, is_final=False),
                    )
            except Exception as exc:
                logger.exception("streaming generation failed")
                await context.abort(grpc.StatusCode.INTERNAL, f"generation failed: {exc}")

            yield pb2.GenerateStreamResponse(
                request_id=request_id,
                token=pb2.TokenDelta(
                    index=0,
                    delta_text="",
                    is_final=True,
                    finish_reason="stop",
                ),
            )

    return LLMInferenceServicer()


async def serve() -> None:
    """启动真实 gRPC/vLLM 服务。"""
    model_path = os.environ.get("MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError(
            "MODEL_PATH 未设置。请指向已获授权且与当前 vLLM 版本兼容的本地模型或 Hub 模型。"
        )

    grpc, pb2, pb2_grpc, engine_args_class, runtime = _load_runtime()
    engine_class, sampling_params_class = runtime
    tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    port = int(os.environ.get("GRPC_PORT", "50051"))

    engine_args = engine_args_class(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "32768")),
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")),
        enable_prefix_caching=True,
    )
    engine = engine_class.from_engine_args(engine_args)

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
        ]
    )
    servicer = _make_servicer(
        grpc=grpc,
        pb2=pb2,
        pb2_grpc=pb2_grpc,
        engine=engine,
        sampling_params_class=sampling_params_class,
    )
    pb2_grpc.add_LLMInferenceServicer_to_server(servicer, server)
    bound_port = server.add_insecure_port(f"[::]:{port}")
    if bound_port == 0:
        raise RuntimeError(f"无法绑定 gRPC 端口 {port}")

    await server.start()
    logger.info("gRPC LLM server listening on :%s (insecure local teaching endpoint)", port)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=10)


if __name__ == "__main__":
    if os.environ.get("GRPC_VLLM_SERVER_RUN") != "1":
        print(
            "[SKIP] Set GRPC_VLLM_SERVER_RUN=1 only after generating protobuf modules "
            "and reviewing MODEL_PATH, GPU capacity, and the local listening port."
        )
        print("OK")
    else:
        asyncio.run(serve())
