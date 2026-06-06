# ---
# chapter: 24
# topic: 云原生部署与工程化
# section: 24.6.5.7 Ollama Cloud + ollama launch (2026 新形态)
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: ollama (Python client)
# run: python 05_ollama_cloud_integration.py
# expected_runtime: 1-5s (or longer if calling real models)
# expected_output: prints both local and cloud chat responses, or mock fallback
# ---
# See: ../tutorial/24_云原生部署与工程化.md §24.6.5.7
# Interview hooks:
#   1. 混合云推理（本地 + Ollama Cloud）的数据合规边界如何划定？
#   2. 用 Bearer Token 调云端 Ollama 时，如何轮转密钥避免被截获？
#   3. ollama.Client 与 openai.OpenAI 客户端的 API 兼容性如何？
"""
Ollama Cloud + 本地混合调用示例
演示如何同时使用本地 ollama 和 Ollama Cloud 推理服务
"""

# Mock 模式兼容：当 ollama 包不可用时使用 mock
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

    class _MockMessage:
        def __init__(self, content):
            self.content = content
            self.role = "assistant"

    class _MockResponse:
        def __init__(self, content):
            self.message = _MockMessage(content)

    class _MockOllamaClient:
        def __init__(self, host=None, headers=None):
            self.host = host or "http://localhost:11434"
            self.headers = headers or {}
        def chat(self, model, messages):
            text = f"[MOCK ollama reply from {self.host}] model={model} prompt={messages[-1]['content']}"
            return _MockResponse(text)

    class _MockModule:
        Client = _MockOllamaClient
    ollama = _MockModule()
    print("[WARN] ollama not installed, using mock client (no real inference)")


def demo_local_and_cloud():
    """演示本地 + Ollama Cloud 双部署调用。"""

    # 本地模型
    client_local = ollama.Client(host="http://ollama.llm-inference:11434")
    resp_local = client_local.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("=== Local Ollama ===")
    print(f"Model: qwen3:8b")
    print(f"Reply: {resp_local.message.content}")
    print()

    # Ollama Cloud 模型（私有数据不出本地的代理）
    # 注：生产中应从环境变量读取 token
    import os
    ollama_cloud_token = os.environ.get("OLLAMA_CLOUD_TOKEN", "MOCK_TOKEN")
    client_cloud = ollama.Client(
        host="https://api.ollama.cloud",
        headers={"Authorization": f"Bearer {ollama_cloud_token}"}
    )
    resp_cloud = client_cloud.chat(
        model="qwen3-480b-cloud",
        messages=[{"role": "user", "content": "Complex reasoning task"}]
    )
    print("=== Ollama Cloud ===")
    print(f"Model: qwen3-480b-cloud")
    print(f"Reply: {resp_cloud.message.content}")


if __name__ == "__main__":
    demo_local_and_cloud()
    print("\nOK")
