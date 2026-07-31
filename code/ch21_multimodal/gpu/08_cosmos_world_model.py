# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.7.1 世界模型 - NVIDIA Cosmos 3 当前接口
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch, current diffusers, cosmos_guardrail (all lazy; real mode only)
# run: python 08_cosmos_world_model.py --mock
# expected_runtime: immediate skip by default; real runtime depends on checkpoint and hardware
# expected_output: [SKIP] unless real GPU mode and COSMOS3_RUN=1 are both explicit
# ---
# See: 21_多模态大模型.md#2171-世界模型world-models
# Official API: https://huggingface.co/docs/diffusers/main/api/pipelines/cosmos3
# Interview hooks:
#   1. 世界模型与通用视频生成模型应如何按证据区分？
#   2. 为什么“生成成功”不能证明物理正确性或机器人闭环安全？
#   3. Cosmos 3 相比早期分立 Predict/Reason/Transfer 路线有什么接口变化？
"""Cosmos 3 当前 Diffusers 接口的条件性真跑示例。

默认 GPU runner 传入 ``--mock``，因此不会访问网络、加载 CUDA 或下载权重。即使启用
``--real-gpu``，仍须显式设置 ``COSMOS3_RUN=1``；这是为了避免把大型模型下载当作普通
离线验收。真跑只生成一张 text-to-image 样例，不验证物理正确性、动作策略或闭环效果。
"""

import json
import os
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu, skip_if_mock


def main() -> None:
    if skip_if_mock(
        "an NVIDIA GPU, current Cosmos 3 Diffusers dependencies, model-license review, "
        "remote weights, and COSMOS3_RUN=1"
    ):
        return

    if os.environ.get("COSMOS3_RUN") != "1":
        print("[SKIP] Real GPU mode selected, but COSMOS3_RUN=1 was not explicitly set.")
        print("Review the current model card, license, guardrail requirements, storage, and VRAM first.")
        return

    require_nvidia_gpu(min_vram_gb=0, min_count=1)

    try:
        import torch
        from diffusers import Cosmos3OmniPipeline
    except ImportError:
        print("[SKIP] Current diffusers with Cosmos3OmniPipeline is not installed.")
        print("Follow the NVIDIA Cosmos 3 / Diffusers installation guide for a compatible environment.")
        return

    model_id = os.environ.get("COSMOS3_MODEL_ID", "nvidia/Cosmos3-Nano")
    output_path = Path(os.environ.get("COSMOS3_OUTPUT", "cosmos3_t2i.jpg")).resolve()
    prompt = json.dumps({"scene": os.environ.get("COSMOS3_SCENE", "A robot arm in a kitchen")})

    print(f"Loading {model_id}; this may download large remote weights.")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    result = pipe(
        prompt=prompt,
        num_frames=1,
        height=720,
        width=1280,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.video[0].save(output_path, format="JPEG", quality=85)

    print(f"Saved one Cosmos 3 text-to-image output: {output_path}")
    print("Boundary: this does not validate physical consistency, action control, or robot safety.")


if __name__ == "__main__":
    main()
    print("OK")
