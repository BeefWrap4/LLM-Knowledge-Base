# ---
# chapter: 29
# topic: Haystack ChatPromptBuilder — 组装多轮对话 + RAG 的 Context
# section: 29.8
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: haystack-ai (optional)
# run: python 05_haystack_chat_pipeline.py
# expected_runtime: <1s (mock)
# ---
#
# See: ../tutorial/29_Context_Engineering.md §29.8
# Cross-refs:
#   - Ch13 Prompt Engineering (System + Few-shot)
#   - Ch15 Agent (Tool schema 注入)
#   - Ch14 RAG (文档注入 messages)
#
# Interview hooks:
#   - "ChatPromptBuilder vs PromptBuilder 区别?" →  ChatPromptBuilder 输出 messages[], 支持 system/user/assistant/tool
#   - "如何注入工具 schema?"                    →  通过 system message 或独立 tool 角色
#   - "多轮对话如何保留 context?"                →  messages[] 累积, 由 checkpointer 持久化

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str  # system | user | assistant | tool
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    meta: dict = field(default_factory=dict)


class MockChatPromptBuilder:
    """模拟 haystack ChatPromptBuilder, 接受 template 中的变量并渲染 messages[]。"""

    def __init__(self, template: list[dict]):
        self.template = template  # [{"role":"system","template":"..."}]

    def run(self, **kwargs) -> dict:
        msgs: list[ChatMessage] = []
        for tpl in self.template:
            content = tpl["template"]
            for k, v in kwargs.items():
                # 仅支持 {{ variable }} 简单替换
                content = content.replace("{{ " + k + " }}", str(v))
            msgs.append(ChatMessage(role=tpl["role"], content=content))
        return {"messages": msgs}


def build_context_engineered_pipeline() -> MockChatPromptBuilder:
    """Context = Instructions + RAG + Tools + History 四维同时注入。"""
    return MockChatPromptBuilder(
        template=[
            {
                "role": "system",
                "template": (
                    "你是一个企业知识助手, 回答须引用文档。\n"
                    "可用工具: {{ tools_list }}\n"
                    "用户偏好: {{ user_preferences }}"
                ),
            },
            {
                "role": "system",
                "template": ("相关文档片段:\n{% for d in documents %}- {{ d }}\n{% endfor %}"),
            },
            {
                "role": "user",
                "template": "{{ question }}",
            },
        ]
    )


def run_demo() -> None:
    builder = build_context_engineered_pipeline()
    result = builder.run(
        tools_list="search_web, query_db, send_email",
        user_preferences="关注科技板块, 风险偏好中等",
        documents=[
            "2026 Q1 财报: 半导体板块同比 +18%",
            "央行 2026/04 降准 25bp",
        ],
        question="当前半导体板块景气度如何?",
    )
    print("=== ChatPromptBuilder 组装的 Context (4 维) ===\n")
    for i, m in enumerate(result["messages"], 1):
        print(f"--- message[{i}] role={m.role} ---")
        print(m.content)
        print()
    # 估算 token
    total = sum(len(m.content) // 2 for m in result["messages"])
    print(f"[估] 总 tokens ≈ {total}")
    print("OK")


if __name__ == "__main__":
    run_demo()
