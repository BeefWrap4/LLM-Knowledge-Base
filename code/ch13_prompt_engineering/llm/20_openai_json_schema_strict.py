# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.5.4 OpenAI JSON Schema 严格模式
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai, pydantic
# run: python 20_openai_json_schema_strict.py
# expected_runtime: 3-10s (real api)
# expected_output: 打印符合 schema 的结构化输出
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.4
# Interview hooks:
# - OpenAI JSON Schema strict 模式与 JSON Mode 的差异？
# - 为何 Pydantic + model_json_schema 是推荐组合？
# - strict/parse 在拒绝或截断时应如何处理？

import os

from pydantic import BaseModel

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class UserInfo(BaseModel):
    name: str
    age: int
    skills: list[str]


def _real_api_ready() -> bool:
    if os.environ.get("LLM_MOCK") != "0":
        print("[SKIP] 离线安全模式：只有显式设置 LLM_MOCK=0 才会调用 OpenAI API")
        print("OK")
        return False
    if OpenAI is None or not os.environ.get("OPENAI_API_KEY"):
        print("[SKIP] 真实调用需要 openai、pydantic 和 OPENAI_API_KEY")
        print("OK")
        return False
    return True


def call_openai_structured(user_text: str):
    if not _real_api_ready():
        return None
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client.responses.parse(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.6"),
        input=[
            {"role": "system", "content": "从用户描述中提取结构化信息。"},
            {"role": "user", "content": user_text},
        ],
        text_format=UserInfo,
    )


if __name__ == "__main__":
    response = call_openai_structured("张伟今年 28 岁，擅长 Python 和 Rust。")
    if response is None:
        raise SystemExit(0)
    if response.output_parsed is None:
        raise RuntimeError(f"未得到结构化结果，status={response.status}")
    data = response.output_parsed
    print(f"[Parsed Data] {data.model_dump()}")
    print(f"[Schema] {UserInfo.model_json_schema()}")
    print("OK")
