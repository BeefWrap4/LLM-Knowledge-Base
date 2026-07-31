# ---
# shared/_mock_fallback.py
# 确定性 mock 响应函数（只供显式 mock/离线场景使用）
# ---
"""
共享的纯函数 mock 响应生成器.

主流程 (`shared.llm_client.UnifiedClient`) 在 ``LLM_MOCK`` 未设置或非 ``0`` 的离线模式下使用它。
真实 API 缺 Key、超时、限流或服务错误必须向调用方抛出，不能降级成貌似成功的 mock。

仅暴露无副作用的纯函数, 避免主流程反向依赖 tests/ 目录.
更复杂的 MockLLM 客户端类位于 tests/_mocks/mock_llm.py (仅 CI/测试可见).

See: tutorial/Ch15_Agent智能体开发, Ch17_大模型评估体系, Ch18_LLM工程框架实战
"""

import hashlib


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


if __name__ == "__main__":
    # Smoke test
    print(deterministic_response("Hello, world!"))
    print(deterministic_response("Hello, world!", max_length=20))
