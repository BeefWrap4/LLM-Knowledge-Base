# ---
# chapter: 27
# topic: LLM 框架与平台选型
# topic_id: llm_frameworks.dify_sdk_usage
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: dify-client (mocked in offline mode)
# run: python 21_dify_sdk_usage.py
# expected_runtime: <1s
# expected_output: SDK method signatures
# ---
# See: ../../../27_LLM框架与平台选型.md
# Interview hooks:
#   1. Dify 的 knowledge base 与向量数据库的关系是什么？
#   2. Dify 的 streaming 与 blocking 模式各适合什么场景？
# Dify 知识库 API 调用示例（Python SDK） - 离线 mock 模式
class DifyClient:
    """Mock DifyClient 模拟 dify_client.DifyClient 的关键方法"""

    def __init__(self, api_key):
        self.api_key = api_key

    def file_upload(self, file, user):
        # 真实场景会发送 HTTP 请求到 Dify 服务
        return {"id": "doc-001", "name": getattr(file, "name", "uploaded")}

    def chat_messages(self, query, user, response_mode="blocking"):
        # 真实场景会通过 SSE 流式返回
        if response_mode == "streaming":
            return iter(
                [
                    {"answer": "公司加班按 1.5 倍工资计算，", "event": "message"},
                    {"answer": "周末 2 倍，法定节假日 3 倍。", "event": "message"},
                ]
            )
        return {
            "answer": "公司加班按 1.5 倍工资计算，周末 2 倍，法定节假日 3 倍。",
            "retriever_resources": [{"document_name": "员工手册.pdf", "content": "加班工资条款..."}],
        }


# 方式1：上传文档到知识库
client = DifyClient(api_key="app-xxxxxxxxxxxx")


class _FileObj:
    name = "企业规章制度.pdf"


uploaded = client.file_upload(file=_FileObj(), user="admin")
print("上传结果:", uploaded)

# 方式2：使用知识库应用进行问答（blocking）
response = client.chat_messages(
    query="公司的加班政策是怎样的？",
    user="employee_001",
    response_mode="blocking",
)
print("blocking 模式回答:", response["answer"])
print("引用:", [r["document_name"] for r in response.get("retriever_resources", [])])

# 方式3：streaming 模式
print("\nstreaming 模式:")
for chunk in client.chat_messages(query="年假", user="employee_001", response_mode="streaming"):
    print(chunk.get("answer", ""), end="", flush=True)
print()

if __name__ == "__main__":
    print("OK")
