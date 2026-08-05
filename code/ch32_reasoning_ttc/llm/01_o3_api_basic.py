# ---
# chapter: 32
# topic: 推理模型与 Test-Time Compute
# topic_id: reasoning_ttc.o3_api_basic
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: openai>=2.51.0,<3
# run: python 01_o3_api_basic.py
# expected_runtime: <2s mock; variable for 3 real API calls
# expected_output: prints 3 effort responses; real API requires LLM_MOCK=0 and a key
# ---
# See: ../../../32_推理模型与Test_Time_Compute.md
# Interview hooks:
#   1. 为什么当前推理工作流优先使用 Responses API，而不是历史 o3-mini 示例？
#   2. reasoning.effort 是行为控制还是严格的 token 预算？
#   3. GPT-5.6 Sol 支持哪些 effort 档位，生产中如何用评测选档？
"""用 GPT-5.6 Sol + Responses API 演示 ``reasoning.effort``。

文件名保留 ``o3`` 仅为兼容既有教程链接；o3/o3-mini 属于历史演进示例，不是本例当前默认。
脚本默认离线 mock；只有显式设置 ``LLM_MOCK=0`` 才会调用真实 API。
"""

import os
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help


def get_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or key == "YOUR_API_KEY":
        raise_with_help(
            "OPENAI_API_KEY 未设置",
            "真实调用需同时设置 `LLM_MOCK=0` 与 `OPENAI_API_KEY`；离线运行请保留默认 mock。",
        )
    return key


def main():
    if os.environ.get("LLM_MOCK", "1").strip() != "0":
        print("=== GPT-5.6 Sol Responses API（离线 mock，默认）===")
        for effort in ("low", "medium", "high"):
            print(f"reasoning.effort={effort}: 9.9 更大（确定性示例，不代表真实 token 或质量）")
        return

    api_key = get_openai_key()

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    print("=== GPT-5.6 Sol + Responses API ===\n")

    # 控制真实调用成本，这里只比较三个代表档位；GPT-5.6 还支持 none/xhigh/max。
    question = "9.11 和 9.9 哪个更大? 详细推理"

    for effort in ["low", "medium", "high"]:
        response = client.responses.create(
            model="gpt-5.6-sol",
            input=question,
            reasoning={"effort": effort},
            max_output_tokens=1024,
        )
        print(f"\n--- reasoning.effort={effort} ---")
        print(response.output_text or "(无文本输出)")
        if response.usage is not None:
            print(
                "usage: "
                f"input={response.usage.input_tokens}, "
                f"output={response.usage.output_tokens}, "
                f"total={response.usage.total_tokens}"
            )


if __name__ == "__main__":
    main()
    print("OK")
