# ---
# tests/_mocks/mock_llm.py
# 确定性 LLM stub — 仅 CI/测试用, 避免真实 API 调用
# ---
"""
Mock LLM 客户端 (CI-only).

此模块位于 tests/_mocks/ 下, 仅在 pytest conftest 自动加载时可见.
主流程 (code/shared/llm_client.py 等) 不应 import 此模块 —
如需确定性响应, 请 from shared._mock_fallback import deterministic_response.

迁移历史:
  原位于 code/shared/mock_llm.py, 因 MockLLM 类属于测试辅助, 不应混入主流程,
  故迁移至 tests/_mocks/mock_llm.py (W1-T5).

See: tutorial/Ch22_Agent基础与工具调用, Ch36_大模型评估基础, Ch27_LLM框架与平台选型
"""

from typing import Any

from shared._mock_fallback import deterministic_response


class MockLLM:
    """Mock LLM 客户端 — drop-in 替代 openai/anthropic client.

    实现统一接口:
        client.chat.completions.create(messages=..., **kwargs)
            → ChatCompletion(message=Choice(content=...))

    真实 LLM 客户端使用时:
        openai.OpenAI().chat.completions.create(...)
    """

    def __init__(self, default_response: str | None = None, deterministic: bool = True):
        self.default_response = default_response
        self.deterministic = deterministic
        self.call_log: list[dict] = []

    @property
    def chat(self):
        return _MockChatNamespace(self)


class _MockChatNamespace:
    def __init__(self, parent: MockLLM):
        self.parent = parent

    @property
    def completions(self):
        return _MockCompletions(self.parent)


class _MockCompletions:
    def __init__(self, parent: MockLLM):
        self.parent = parent

    def create(self, model: str = "mock-model", messages: list = None, **kwargs) -> Any:
        messages = messages or []
        prompt = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        self.parent.call_log.append({"model": model, "messages": messages, "kwargs": kwargs})
        text = self.parent.default_response or deterministic_response(prompt)

        # 返回 shape 类似 OpenAI 响应
        return _MockChatCompletion(text)


class _MockChoice:
    def __init__(self, text: str):
        self.message = type("Message", (), {"content": text})()


class _MockChatCompletion:
    def __init__(self, text: str):
        self.id = "mock-completion"
        self.model = "mock-model"
        self.choices = [_MockChoice(text)]
        self.usage = type("Usage", (), {"total_tokens": len(text.split())})()


if __name__ == "__main__":
    # Smoke test
    client = MockLLM()
    response = client.chat.completions.create(
        model="gpt-mock",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print("Mock response:", response.choices[0].message.content)
    print("Calls logged:", len(client.call_log))
