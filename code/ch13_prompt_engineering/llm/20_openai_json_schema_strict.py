# ---
# chapter: 13
# topic: Prompt Engineering
# section: 13.7.4 OpenAI JSON Schema 严格模式
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: openai (可选), pydantic
# run: python 20_openai_json_schema_strict.py
# expected_runtime: <1s (mock) / 3-10s (real api)
# expected_output: 打印符合 schema 的结构化输出
# ---
# See: ../tutorial/13_Prompt_Engineering.md#13.7.4
# Interview hooks:
# - OpenAI JSON Schema strict 模式与 JSON Mode 的差异？
# - 为何 Pydantic + model_json_schema 是推荐组合？
# - strict=True 是否影响延迟？(轻微增加)

import os
import json

USE_MOCK = os.environ.get("USE_REAL_API") != "1"

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = object  # 极简兼容


class UserInfo(BaseModel):
    name: str
    age: int
    skills: list[str]


class _MockChoice:
    class _Msg:
        content = json.dumps({"name": "张伟", "age": 28, "skills": ["Python", "Rust"]},
                             ensure_ascii=False)
    message = _Msg()


class _MockResp:
    choices = [_MockChoice()]


def call_openai_structured(user_text: str):
    if USE_MOCK:
        return _MockResp()

    # OpenAI JSON Mode：response_format={"type": "json_schema", "schema": {...}}
    from openai import OpenAI
    client = OpenAI()
    return client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "从用户描述中提取结构化信息。"},
            {"role": "user", "content": user_text}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "user_info",
                "schema": UserInfo.model_json_schema(),
                "strict": True  # 严格模式：100% 符合 schema
            }
        }
    )


if __name__ == "__main__":
    response = call_openai_structured("张伟今年 28 岁，擅长 Python 和 Rust。")

    # 输出 100% 符合 schema，可直接 parse
    data = json.loads(response.choices[0].message.content)
    # data = {"name": "张伟", "age": 28, "skills": ["Python", "Rust"]}
    print(f"[Parsed Data] {data}")
    print(f"[Schema] {UserInfo.model_json_schema()}")
    print("OK")
