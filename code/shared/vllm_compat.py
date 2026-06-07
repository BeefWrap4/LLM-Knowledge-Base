# ---
# shared/vllm_compat.py
# vLLM Docker escape hatch — Windows 友好兼容层
# ---
"""
vLLM Docker escape hatch — 让 Windows 用户也能跑 vLLM 例子.

3 种模式 (按优先级):
  1. Docker server: VLLM_BASE_URL 设了 → 走 OpenAI 协议连 Docker vLLM
  2. 真 vllm: VLLM_BASE_URL 未设 + vllm 完整 (含 _C) → 走 vllm.LLM
  3. 友好抛错: vllm._C 缺失 → RuntimeError 提示用 Docker escape hatch

7 个 ch25 例子仅需换 1 行 import:

    from vllm import LLM, SamplingParams
    # →
    from shared.vllm_compat import LLM, SamplingParams

主体代码 0 改动, API 兼容.
"""
import os
from typing import Any, AsyncIterator, List, Optional, Union

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
USE_DOCKER = bool(VLLM_BASE_URL)


def _get_model_id() -> str:
    """Docker 容器内默认 /root/.cache/huggingface/, 可通过 VLLM_MODEL_ID 覆盖."""
    return os.environ.get(
        "VLLM_MODEL_ID",
        "/root/.cache/huggingface/Qwen2.5-0.5B-Instruct",
    )


# ── SamplingParams / EngineArgs (纯 dataclass, 双模式通用) ──

class SamplingParams:
    """兼容 vllm.SamplingParams (本例子只用 temperature + max_tokens)."""
    def __init__(self, temperature: float = 0.7, max_tokens: int = 64,
                 top_p: float = 1.0, top_k: int = -1,
                 stop: Optional[List[str]] = None, **kwargs: Any):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop or []
        self.extra = kwargs


class EngineArgs:
    """兼容 vllm.EngineArgs (无 _C 校验, 用于配置展示)."""
    def __init__(self, model: str, max_num_seqs: int = 8,
                 gpu_memory_utilization: float = 0.5, max_model_len: int = 2048,
                 enforce_eager: bool = True, tensor_parallel_size: int = 1,
                 enable_expert_parallel: bool = False, **kwargs: Any):
        self.model = model
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_expert_parallel = enable_expert_parallel
        self.extra = kwargs


class AsyncEngineArgs(EngineArgs):
    """兼容 vllm.AsyncEngineArgs (同 EngineArgs 字段)."""
    pass


# ── Mock config objects (Docker 模式用于演示打印, 不真跑模型) ──

class _MockCacheConfig:
    def __init__(self, gpu_memory_utilization: float = 0.5):
        self.block_size = 16
        self.gpu_memory_utilization = gpu_memory_utilization


class _MockSchedulerConfig:
    def __init__(self, max_num_seqs: int = 8):
        self.max_num_seqs = max_num_seqs


class _MockParallelConfig:
    def __init__(self, tensor_parallel_size: int = 1, enable_expert_parallel: bool = False):
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_expert_parallel = enable_expert_parallel
        self.data_parallel_size = 1


class _MockVllmConfig:
    def __init__(self, engine_args: EngineArgs):
        self.cache_config = _MockCacheConfig(gpu_memory_utilization=engine_args.gpu_memory_utilization)
        self.scheduler_config = _MockSchedulerConfig(max_num_seqs=engine_args.max_num_seqs)
        self.parallel_config = _MockParallelConfig(
            tensor_parallel_size=engine_args.tensor_parallel_size,
            enable_expert_parallel=engine_args.enable_expert_parallel,
        )


class _MockLLMEngine:
    def __init__(self, engine_args: EngineArgs):
        self.vllm_config = _MockVllmConfig(engine_args)


# ── Mock RequestOutput (Docker 模式) ──

class _Output:
    def __init__(self, text: str, finish_reason: str = "stop", token_ids: Optional[List[int]] = None):
        self.text = text
        self.finish_reason = finish_reason
        self.token_ids = token_ids or list(range(len(text.split())))


class _RequestOutput:
    def __init__(self, request_id: str, text: str, finished: bool = True, token_ids: Optional[List[int]] = None):
        self.request_id = request_id
        self.outputs = [_Output(text, token_ids=token_ids)]
        self.finished = finished


def _approx_token_count(text: str) -> int:
    return max(1, len(text.split()))


# ── 模式 1: Docker server (OpenAI 协议) ──

class _CompatLLM:
    """OpenAI 协议客户端, 模拟 vllm.LLM.generate() 同步接口."""

    def __init__(self, model: Optional[str] = None, **kwargs: Any):
        from openai import OpenAI  # type: ignore
        self.client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="EMPTY")
        self.model = model or _get_model_id()
        self.llm_engine = _MockLLMEngine(EngineArgs(model=self.model, **kwargs))
        print(f"  [vllm_compat] Docker server: {VLLM_BASE_URL}, model={self.model}")

    def generate(self, prompts: Union[str, List[str]],
                 sampling_params: Optional[SamplingParams] = None,
                 **kwargs: Any) -> List[_RequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        sp = sampling_params or SamplingParams()
        results: List[_RequestOutput] = []
        for i, prompt in enumerate(prompts):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=sp.temperature, max_tokens=sp.max_tokens,
                top_p=sp.top_p, stop=sp.stop or None,
            )
            text = (resp.choices[0].message.content or "") if resp.choices else ""
            results.append(_RequestOutput(
                request_id=f"r{i}", text=text,
                token_ids=list(range(_approx_token_count(text))),
            ))
        return results


class _CompatAsyncLLMEngine:
    """OpenAI 协议客户端, 模拟 vllm.AsyncLLMEngine.generate() 异步流式接口."""

    @classmethod
    def from_engine_args(cls, args: EngineArgs, **kwargs: Any) -> "_CompatAsyncLLMEngine":
        return cls(model=args.model, **kwargs)

    def __init__(self, model: Optional[str] = None, **kwargs: Any):
        from openai import AsyncOpenAI  # type: ignore
        self.client = AsyncOpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="EMPTY")
        self.model = model or _get_model_id()
        self.llm_engine = _MockLLMEngine(EngineArgs(model=self.model, **kwargs))
        print(f"  [vllm_compat] Docker server (async): {VLLM_BASE_URL}, model={self.model}")

    async def generate(self, prompt: str,
                       sampling_params: Optional[SamplingParams] = None,
                       request_id: str = "default", **kwargs: Any) -> AsyncIterator[_RequestOutput]:
        sp = sampling_params or SamplingParams()
        full_text = ""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=sp.temperature, max_tokens=sp.max_tokens,
            top_p=sp.top_p, stop=sp.stop or None, stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content
                yield _RequestOutput(request_id=request_id, text=full_text, finished=False,
                                     token_ids=list(range(_approx_token_count(full_text))))
        yield _RequestOutput(request_id=request_id, text=full_text, finished=True,
                             token_ids=list(range(_approx_token_count(full_text))))


# ── 模式 2: 真 vllm (无 _C 校验, 延迟到首次调用时报错) ──

def _try_import_real_vllm():
    """尝试 import 真 vllm. 失败 (含 _C 缺失) 时 raise_with_help."""
    try:
        import vllm  # type: ignore
        import vllm._C  # type: ignore  # noqa  # 触发 _C 加载
        return vllm
    except (ImportError, ModuleNotFoundError, OSError) as e:
        from shared._error_helper import raise_with_help
        raise_with_help(
            f"vllm._C 编译扩展不可用: {e}",
            "修复路径: 1) Linux + `pip install vllm`; 2) WSL2 + 同上; "
            "3) Docker escape hatch: `export VLLM_BASE_URL=http://localhost:8000` + "
            "`make vllm-server-start` (参考 code/README.md).",
        )


# ── 公开 API: 自动调度 ──

def LLM(model: Optional[str] = None, **kwargs: Any):
    """兼容 vllm.LLM, 自动调度 Docker / 真 vllm."""
    if USE_DOCKER:
        return _CompatLLM(model=model, **kwargs)
    vllm = _try_import_real_vllm()
    return vllm.LLM(model=model, **kwargs)


class AsyncLLMEngine:
    """兼容 vllm.AsyncLLMEngine, 自动调度."""

    @classmethod
    def from_engine_args(cls, args: EngineArgs, **kwargs: Any):
        if USE_DOCKER:
            return _CompatAsyncLLMEngine.from_engine_args(args, **kwargs)
        vllm = _try_import_real_vllm()
        return vllm.AsyncLLMEngine.from_engine_args(args, **kwargs)
