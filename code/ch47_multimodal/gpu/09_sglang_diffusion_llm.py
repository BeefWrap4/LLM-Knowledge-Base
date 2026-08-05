# ---
# chapter: 49
# topic: 世界模型、VLA 与具身智能
# topic_id: multimodal.sglang_diffusion_llm
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: httpx
# run: SGLANG_DIFFUSION_RUN=1 CH21_SGLANG_MODEL=<served-model> python 09_sglang_diffusion_llm.py
# expected_runtime: depends on the explicitly managed local service
# expected_output: response from the configured SGLang model
# ---
# See: ../../../49_世界模型VLA与具身智能.md
# Interview hooks:
#   1. 扩散语言模型与自回归生成的解码依赖有何差异？
#   2. 服务端支持某个模型，为什么不能只由客户端参数推断？
#   3. 如何记录服务端 revision、启动参数、延迟和输出质量？

import os
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import skip_if_mock


def main() -> None:
    if skip_if_mock("an explicitly managed SGLang endpoint and served model"):
        return
    if os.environ.get("SGLANG_DIFFUSION_RUN") != "1":
        print("[SKIP] Set SGLANG_DIFFUSION_RUN=1 only after starting and identifying the server.")
        return

    model = os.environ.get("CH21_SGLANG_MODEL", "").strip()
    if not model:
        raise RuntimeError("CH21_SGLANG_MODEL must equal the model name reported by the server")
    base_url = os.environ.get("CH21_SGLANG_BASE_URL", "http://127.0.0.1:30000/v1").rstrip("/")

    import httpx

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "用一句话解释快速排序的平均复杂度。"}],
        "max_tokens": 64,
        "temperature": 0,
    }
    response = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    if not content:
        raise RuntimeError("SGLang returned an empty assistant message")
    print(f"model={body.get('model', model)}")
    print(content)


if __name__ == "__main__":
    main()
    print("OK")
