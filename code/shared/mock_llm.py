# ---
# shared/mock_llm.py
# 确定性 LLM stub — 测试用, 避免真实 API 调用
# ---
"""
See: tutorial/Ch15_Agent智能体开发, Ch17_大模型评估体系, Ch18_LLM工程框架实战
"""
import hashlib
import re
from typing import Any, Optional


def deterministic_response(prompt: str, max_length: int = 64) -> str:
    """基于 prompt hash 的确定性响应 — 相同 prompt 永远返回相同输出."""
    h = hashlib.md5(prompt.encode()).hexdigest()[:8]
    words = [
        "这是一个确定性 mock 响应。",
        f"prompt hash: {h}",
        "真实 LLM 会返回更有意义的答案。",
        "本 mock 用于 smoke testing。",
    ]
    response = " ".join(words)
    return response[:max_length]


class MockLLM:
    """Mock LLM 客户端 — drop-in 替代 openai/anthropic client.

    实现统一接口:
        client.chat.completions.create(messages=..., **kwargs)
            → ChatCompletion(message=Choice(content=...))

    真实 LLM 客户端使用时:
        openai.OpenAI().chat.completions.create(...)
    """

    def __init__(self, default_response: Optional[str] = None, deterministic: bool = True):
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
        self.parent.call_log.append(
            {"model": model, "messages": messages, "kwargs": kwargs}
        )
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
    print("OK")
