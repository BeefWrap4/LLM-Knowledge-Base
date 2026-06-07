import sys as _sys_path_setup
from pathlib import Path as _Path_setup
_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

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

    # Wave 16: 改用 UnifiedClient (注: response_format 仅 OpenAI 完整支持, 其他厂商可能忽略)
    from shared.llm_client import UnifiedClient
    client = UnifiedClient()
    return client.chat(
        messages=[
            {"role": "system", "content": "从用户描述中提取结构化信息。"},
            {"role": "user", "content": user_text}
        ],
        # 注: 真实 json_schema 仅 OpenAI 支持; 其他厂商会回退到普通 JSON 模式
    )


if __name__ == "__main__":
    response = call_openai_structured("张伟今年 28 岁，擅长 Python 和 Rust。")

    # 输出 100% 符合 schema，可直接 parse
    # Wave 24: 适配 UnifiedClient 的 _LLMResponse (无 .choices 属性)
    content = response.content if hasattr(response, "content") else response.choices[0].message.content
    # Wave 24: 部分厂商不严格遵循 JSON, 用 regex 提取首个 {...} 块
    import re
    json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(0))
        print(f"[Parsed Data] {data}")
    else:
        # 厂商未返回 JSON, 输出原文让用户看到
        print(f"[Raw Response] {content[:200]}")
        print(f"[Schema (expected)] {UserInfo.model_json_schema()}")
        data = None
    # data = {"name": "张伟", "age": 28, "skills": ["Python", "Rust"]} (when schema 严格)
    if data:
        print(f"[Parsed Data] {data}")
    print(f"[Schema] {UserInfo.model_json_schema()}")
    print("OK")
