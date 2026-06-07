# W5 推理引擎 GPU 实现计划 — Real API Code

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 把 `ch25_inference_engines/gpu/` 12 个文件从 mock / 简化类改为真实调用 vLLM / SGLang / TensorRT-LLM 真实引擎。

**前置依赖：** W1 + W2 + W3 + W4 完成。

**目标硬件：** NVIDIA GPU 24GB+

---

## 文件清单

### 修改 / 重命名

- `code/ch25_inference_engines/gpu/01_paged_attention_block_manager.py`
- `code/ch25_inference_engines/gpu/02_continuous_batching_scheduler.py`
- `code/ch25_inference_engines/gpu/03_radix_attention_prefix_tree.py`
- `code/ch25_inference_engines/gpu/04_pd_disaggregation.py`
- `code/ch25_inference_engines/gpu/05_kv_cache_memory_calculator.py`（仅审）
- `code/ch25_inference_engines/gpu/06_speculative_decoding.py`
- `code/ch25_inference_engines/gpu/07_moe_expert_parallel.py`
- `code/ch25_inference_engines/gpu/08_tensorrt_llm_build_mock.py` → **改名** `08_tensorrt_llm_build.py`
- `code/ch25_inference_engines/gpu/09_fp4_quantization_ladder.py`
- `code/ch25_inference_engines/gpu/10_vllm_async_engine_client.py` → **改名** `10_vllm_async_engine.py`
- `code/ch25_inference_engines/gpu/11_slo_ttft_tpot_monitor.py`
- `code/ch25_inference_engines/gpu/12_engine_selection_decision_tree.py`

---

## 任务 1：`10_vllm_async_engine.py` — 删除 `MockAsyncLLMEngine`

- [ ] **步骤 1：读 `code/ch25_inference_engines/gpu/10_vllm_async_engine_client.py` 现状**

- [ ] **步骤 2：git mv 改名**

```bash
cd code
git mv ch25_inference_engines/gpu/10_vllm_async_engine_client.py \
       ch25_inference_engines/gpu/10_vllm_async_engine.py
```

- [ ] **步骤 3：删除 `MockAsyncLLMEngine` / `Mock RequestOutput` / `Mock SamplingParams` 类**

- [ ] **步骤 4：替换为真实 vLLM**

```python
# 10_vllm_async_engine.py
import asyncio
import sys
from pathlib import Path
_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu

def check_hardware():
    require_nvidia_gpu(min_vram_gb=24)

async def main():
    check_hardware()
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    
    model_path = "code/models/Qwen2.5-7B-Instruct"
    if not Path(model_path).exists():
        from shared._error_helper import raise_with_help
        raise_with_help(f"需要模型 {model_path}", "运行 `make download-models-llm`.")
    
    args = AsyncEngineArgs(
        model=model_path,
        max_num_seqs=64,
        gpu_memory_utilization=0.9,
    )
    engine = AsyncLLMEngine.from_engine_args(args)
    sampling = SamplingParams(temperature=0.7, max_tokens=64)
    
    print("Streaming:")
    async for out in engine.generate("讲个笑话", sampling, request_id="r-1"):
        print(out.outputs[0].text, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **步骤 5：跑（NVIDIA 24GB+ 上）**

```bash
cd code
python ch25_inference_engines/gpu/10_vllm_async_engine.py
```

预期：真实加载 Qwen2.5-7B 并流式生成。

- [ ] **步骤 6：Commit**

```bash
git add code/ch25_inference_engines/gpu/10_vllm_async_engine.py
git rm code/ch25_inference_engines/gpu/10_vllm_async_engine_client.py
git commit -m "W5 ch25/10: real vllm.AsyncLLMEngine (rename + delete mock)"
```

---

## 任务 2：`08_tensorrt_llm_build.py` — 删除 mock build

模式同任务 1：git mv 改名 + 真实 `subprocess.run(["trtllm-build", ...])`。

- [ ] **步骤 1-6**：同任务 1 模式

---

## 任务 3：`01-04, 06, 07, 09, 11, 12` — 真实化 vLLM 内部类

| 文件 | 改造 |
|------|------|
| `01` paged attention | `from vllm.block_manager import BlockManager`（不启 engine） |
| `02` continuous batching | `from vllm.scheduler import Scheduler` |
| `03` radix attention | `from vllm.prefix_caching import Tree` |
| `04` PD disagg | `from vllm.disagg import DisaggEngine` |
| `06` speculative | `from vllm.spec_decode import SpeculativeDecoder` |
| `07` MoE expert parallel | `from vllm.expert_parallel import ExpertParallel` |
| `09` FP4 quantization | `from transformers import BitsAndBytesConfig` 真实量化 |
| `11` SLO 监控 | 真实 `prometheus_client.start_http_server` |
| `12` 引擎选择 | 决策函数（纯逻辑，无需改） |

---

## 任务 4：手测（NVIDIA 24GB+ 上）

```bash
cd code
python ch25_inference_engines/gpu/10_vllm_async_engine.py 2>&1 | head -20
python ch25_inference_engines/gpu/05_kv_cache_memory_calculator.py  # 纯计算无需 GPU
python ch25_inference_engines/gpu/12_engine_selection_decision_tree.py  # 纯逻辑
```

---

## 任务 5：教程 25 章节同步更新

加 vLLM 真实启动 + 硬件需求段；删 mock 描述。

---

## 任务 6：Commit 收尾

```bash
git add -A
git commit -m "W5 ch25 inference engines: all 12 files use real vLLM/TRT-LLM"
```

---

## W5 验收清单

- [ ] `10_vllm_async_engine.py` 真实加载 Qwen2.5-7B 并流式生成
- [ ] `08_tensorrt_llm_build.py` 真实 `trtllm-build`
- [ ] 01-04, 06, 07, 09, 11 用 vLLM / transformers / prometheus_client 真实类
- [ ] 05 (纯计算), 12 (纯逻辑) 仅审不改
- [ ] 教程 25 章节更新
- [ ] 缺 GPU / 缺权重时明确 RuntimeError
