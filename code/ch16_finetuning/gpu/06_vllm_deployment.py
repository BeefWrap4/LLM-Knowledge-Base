# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.6.2 vLLM 部署实战
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: vllm, fastapi, uvicorn, pydantic
# run: python 06_vllm_deployment.py --mock
# expected_runtime: <5s for mock / 启动后持续服务
# expected_output: mock 模式演示 vLLM 三种使用方式（脚本 / FastAPI / Docker）
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.6.2
# Interview hooks:
#   1. vLLM 的 PagedAttention + Continuous Batching 如何实现高吞吐？
#   2. gpu_memory_utilization=0.85 的含义？为什么不能设到 1.0？
#   3. tensor_parallel_size 与 Pipeline Parallelism 的使用场景差异？

"""
vLLM 部署大模型服务 —— 三种使用方式
    1. 脚本直接推理
    2. FastAPI 服务封装（OpenAI 兼容 API）
    3. Docker 容器化部署
"""

import os
import argparse


MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"


def mock_vllm_deployment():
    """无 GPU / 无 vllm 环境下的 mock 演示"""
    print("[MOCK] vLLM 三种使用方式概览")
    print()
    print("=" * 60)
    print("方式1: 脚本直接推理")
    print("=" * 60)
    print("""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=512,
    )

    prompts = ["请介绍机器学习", "什么是深度学习?"]
    outputs = llm.generate(prompts, sampling_params)
    """)
    print()
    print("=" * 60)
    print("方式2: FastAPI 服务封装（见下方代码）")
    print("=" * 60)
    print()
    print("=" * 60)
    print("方式3: Docker 部署")
    print("=" * 60)
    print("""
    FROM vllm/vllm-openai:latest
    ENV MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    EXPOSE 8000
    ENTRYPOINT python -m vllm.entrypoints.openai.api_server \\
        --model ${MODEL_NAME} \\
        --tensor-parallel-size 1 \\
        --gpu-memory-utilization 0.85 \\
        --max-model-len 8192 \\
        --port 8000

    # 调用
    # import openai
    # client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    # response = client.chat.completions.create(
    #     model="Qwen/Qwen2.5-7B-Instruct",
    #     messages=[{"role": "user", "content": "Hello!"}],
    # )
    """)
    print()
    print("OK")


def real_vllm_deployment():
    """真实 vLLM 部署（需 GPU + vllm）"""
    from vllm import LLM, SamplingParams
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    # ========== 方式1: 脚本直接推理 ==========
    def vllm_direct_inference():
        llm = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
        )

        sampling_params = SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=512,
        )

        prompts = [
            "请用一句话介绍机器学习。",
            "什么是深度学习？",
            "Transformer 架构的核心思想是什么？",
        ]

        outputs = llm.generate(prompts, sampling_params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Output: {output.outputs[0].text}\n")

    vllm_direct_inference()

    # ========== 方式2: FastAPI 服务封装 ==========
    app = FastAPI(title="LLM Inference Service")
    llm_engine = None

    class InferenceRequest(BaseModel):
        prompt: str
        temperature: float = 0.7
        top_p: float = 0.9
        max_tokens: int = 512

    class InferenceResponse(BaseModel):
        text: str
        usage: dict

    @app.on_event("startup")
    def load_model():
        nonlocal llm_engine
        llm_engine = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
        )
        print("模型加载完成")

    @app.post("/v1/chat/completions", response_model=InferenceResponse)
    async def chat_completion(request: InferenceRequest):
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        outputs = llm_engine.generate(request.prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text
        input_tokens = len(outputs[0].prompt_token_ids)
        output_tokens = len(outputs[0].outputs[0].token_ids)
        return InferenceResponse(
            text=generated_text,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )

    @app.get("/health")
    def health():
        return {"status": "healthy", "model_loaded": llm_engine is not None}

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or MOCK_MODE:
        mock_vllm_deployment()
    else:
        real_vllm_deployment()
