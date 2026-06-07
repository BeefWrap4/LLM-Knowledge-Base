# ---
# chapter: 24
# topic: 云原生部署与工程化
# section: 24.4.3 gRPC 服务端实现
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: grpcio, grpcio-tools, vllm, protobuf
# run: python 02_grpc_vllm_server.py
# expected_runtime: 60-180s (model load) + blocking
# expected_output: gRPC LLM server listening on :50051
# ---
# See: ../tutorial/24_云原生部署与工程化.md §24.4.3
# Interview hooks:
#   1. gRPC Server Streaming 与 SSE 的性能差异主要来自哪里？
#   2. 为什么大模型推理需要 keepalive 配置，keepalive_time_ms 该如何选？
#   3. Protobuf 与 JSON 在大模型消息体积上的差异如何量化？


# === Multi-GPU / heavy model guard (auto-added) ===
import sys as _sys
import os as _os
_NGPU = _os.environ.get("WORLD_SIZE", "1")
if _NGPU == "1" and not _os.environ.get("FORCE_GPU_RUN"):
    print(f"[SKIP] {{__file__}}: 需多卡 (WORLD_SIZE>1) 或真实模型权重, 用 torchrun 或设置 FORCE_GPU_RUN=1")
    _sys.exit(0)
"""
gRPC 大模型推理服务端 —— 使用 vLLM AsyncEngine

需要先编译 proto:
    python -m grpc_tools.protoc -I protos --python_out=. \
        --grpc_python_out=. protos/llm_inference.proto
"""

import asyncio
import time
import logging
from concurrent import futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grpc-llm-server")

# ====== Mock protobuf 模块（用于无 proto 编译环境的演示） ======
class _MockMessage:
    """模拟 Protobuf 消息对象。"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def __repr__(self):
        return f"_MockMessage({self.__dict__})"


class _MockSamplingParams:
    def __init__(self, **kwargs):
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.top_p = kwargs.get('top_p', 1.0)
        self.top_k = kwargs.get('top_k', -1)


# 创建 proto 模块的占位
class _MockModule:
    def __init__(self):
        self.GenerateRequest = _MockMessage
        self.GenerateResponse = _MockMessage
        self.GenerateStreamResponse = _MockMessage
        self.Completion = _MockMessage
        self.TokenUsage = _MockMessage
        self.TokenDelta = _MockMessage
        self.ChatMessage = _MockMessage
        self.SamplingParams = _MockSamplingParams
    def LLMInferenceServicer(self):
        class _Base:
            pass
        return _Base


# 尝试导入真实 protobuf；失败则使用 Mock
try:
    import grpc
    import llm_inference_pb2 as pb2
    import llm_inference_pb2_grpc as pb2_grpc
    HAS_PROTO = True
except ImportError:
    HAS_PROTO = False
    grpc = None
    pb2 = _MockModule()
    pb2_grpc = _MockModule()
    logger.warning("gRPC/proto not available — running in MOCK mode (no real server)")

try:
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    def random_uuid():
        return f"mock-{int(time.time()*1000)}"
    logger.warning("vLLM not available — using mock SamplingParams")


# ====== Mock vLLM Engine（无 GPU 环境） ======
class MockAsyncEngine:
    """无 vLLM 时的 Mock 推理引擎。"""
    async def generate(self, prompt, sampling_params, request_id):
        tokens = ["Hello", " from", " gRPC", " streaming", "!"]
        text = ""
        for tok in tokens:
            await asyncio.sleep(0.05)
            text += tok
            out = _MockMessage(text=text, token_ids=text.split())
            yield _MockMessage(outputs=[out], prompt_token_ids=prompt.split() if isinstance(prompt, str) else [])


# ====== gRPC Servicer ======
class LLMInferenceServicer:
    """LLM 推理 gRPC 服务实现。

    注：实际生产中继承 pb2_grpc.LLMInferenceServicer 并重写方法；
    这里以动态分派实现，便于 mock 模式。
    """

    def __init__(self, engine):
        self.engine = engine

    async def Generate(self, request, context):
        """单次生成（Unary RPC）"""
        if HAS_VLLM:
            sp = request.sampling_params
            sampling_params = SamplingParams(
                temperature=sp.temperature,
                max_tokens=sp.max_tokens,
                top_p=sp.top_p,
                top_k=sp.top_k,
            )
        else:
            sp = request.sampling_params if hasattr(request, 'sampling_params') else _MockSamplingParams()
            sampling_params = sp

        # 构建 Prompt
        prompt = self._build_prompt(request.messages)

        # 推理
        request_id = getattr(request, 'request_id', None) or random_uuid()
        final_output = None
        async for result in self.engine.generate(
            prompt, sampling_params, request_id
        ):
            final_output = result

        if final_output is None:
            if HAS_PROTO:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Generation returned no output")
            return pb2.GenerateResponse()

        return pb2.GenerateResponse(
            request_id=request_id,
            completion=[
                pb2.Completion(
                    index=0,
                    text=final_output.outputs[0].text,
                    finish_reason="stop",
                )
            ],
            usage=pb2.TokenUsage(
                prompt_tokens=len(final_output.prompt_token_ids),
                completion_tokens=len(final_output.outputs[0].token_ids),
                total_tokens=(
                    len(final_output.prompt_token_ids) +
                    len(final_output.outputs[0].token_ids)
                ),
            ),
        )

    async def GenerateStream(self, request, context):
        """流式生成（Server Streaming RPC）"""
        if HAS_VLLM:
            sp = request.sampling_params
            sampling_params = SamplingParams(
                temperature=sp.temperature,
                max_tokens=sp.max_tokens,
                top_p=sp.top_p,
            )
        else:
            sp = getattr(request, 'sampling_params', _MockSamplingParams())
            sampling_params = sp

        prompt = self._build_prompt(request.messages)
        request_id = getattr(request, 'request_id', None) or random_uuid()

        async for result in self.engine.generate(
            prompt, sampling_params, request_id
        ):
            yield pb2.GenerateStreamResponse(
                request_id=request_id,
                token=pb2.TokenDelta(
                    index=0,
                    delta_text=result.outputs[0].text,
                    is_final=False,
                ),
            )

        # 最终完成信号
        yield pb2.GenerateStreamResponse(
            request_id=request_id,
            token=pb2.TokenDelta(
                index=0,
                delta_text="",
                is_final=True,
                finish_reason="stop",
            ),
        )

    @staticmethod
    def _build_prompt(messages) -> str:
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)


async def serve():
    """启动 gRPC LLM 推理服务。"""

    # 初始化 vLLM Engine 或 Mock
    if HAS_VLLM:
        engine_args = AsyncEngineArgs(
            model="/models/Qwen2.5-72B-Instruct-AWQ",
            tensor_parallel_size=4,
            max_model_len=32768,
            gpu_memory_utilization=0.95,
            enable_prefix_caching=True,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
    else:
        engine = MockAsyncEngine()
        logger.info("Using MockAsyncEngine (no vLLM)")

    if HAS_PROTO:
        # 启动真实 gRPC 服务
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
            ],
        )

        servicer = LLMInferenceServicer(engine)
        # 实际生产中：pb2_grpc.add_LLMInferenceServicer_to_server(servicer, server)
        # 这里动态添加以兼容 Mock
        rpc_method_handlers = {
            "Generate": grpc.unary_unary_rpc_method_handler(servicer.Generate),
            "GenerateStream": grpc.unary_stream_rpc_method_handler(servicer.GenerateStream),
        }
        generic_handler = grpc.method_service_handler("llm.inference.v1.LLMInference", rpc_method_handlers)
        server.add_generic_rpc_handlers((generic_handler,))

        server.add_insecure_port("[::]:50051")
        await server.start()
        print("gRPC LLM Inference Server started on port 50051")
        await server.wait_for_termination()
    else:
        # Mock 模式：仅做一次往返调用演示
        print("MOCK MODE: gRPC server would listen on :50051")
        print("Building mock request ...")
        req = _MockMessage(
            messages=[_MockMessage(role="user", content="Hello")],
            sampling_params=_MockSamplingParams(temperature=0.7, max_tokens=64),
            request_id="demo-1",
        )
        resp = await LLMInferenceServicer(engine).Generate(req, context=None)
        print(f"Unary response: {resp}")
        print("Streaming response:")
        async for chunk in LLMInferenceServicer(engine).GenerateStream(req, context=None):
            print(f"  - {chunk}")
        print("OK")


if __name__ == "__main__":
    asyncio.run(serve())