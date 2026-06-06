# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.1.2 Chain 概念与类型 - RouterChain
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langchain, langchain-openai
# run: python 04_router_chain.py
# expected_runtime: <1s (mock mode)
# expected_output: routed expert answers
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.1.2
# Interview hooks:
#   1. RouterChain 的路由决策是如何实现的？LLM Router 有什么优缺点？
#   2. MultiPromptChain 中 prompt_infos 的 description 有什么作用？
# RouterChain 简化示例：本地模拟路由 + 多个专家 Prompt


# === Optional dependency guard (auto-added) ===
import sys as _sys
try:
    from langchain.chains import ConversationChain
    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    print("OK")
    _sys.exit(0)
from langchain.chains.router import MultiPromptChain
from langchain_core.prompts import PromptTemplate

class _MockChatModel:
    def invoke(self, msgs):
        last = msgs[-1].content if hasattr(msgs[-1], 'content') else str(msgs[-1])
        if "物理" in last or "量子" in last:
            text = "[物理专家] 量子纠缠是两个粒子在空间分离后仍保持关联的现象。"
        elif "数学" in last or "证明" in last:
            text = "[数学专家] 用严谨的数学语言回答..."
        elif "Python" in last or "代码" in last or "排序" in last:
            text = "[编程专家] 快速排序示例：\n```python\ndef qs(a):\n    if len(a)<=1: return a\n    p=a[0]; return qs([x for x in a[1:] if x<p])+[p]+qs([x for x in a[1:] if x>=p])\n```"
        else:
            text = "[通用] 让我想想..."
        class _R: content = text
        return _R()

llm = _MockChatModel()

# 定义不同专业的提示词模板
physics_template = """你是一位物理学专家。请专业地回答以下问题：
{input}"""

math_template = """你是一位数学专家。请用严谨的数学语言回答：
{input}"""

coding_template = """你是一位资深程序员。请用代码示例回答问题：
{input}"""

prompt_infos = [
    {"name": "physics", "description": "适合回答物理问题", "prompt_template": physics_template},
    {"name": "math", "description": "适合回答数学问题", "prompt_template": math_template},
    {"name": "coding", "description": "适合回答编程问题", "prompt_template": coding_template},
]

# 自动路由：LLM 根据问题内容选择最合适的专家
router_chain = MultiPromptChain.from_prompts(
    llm=llm,
    prompt_infos=prompt_infos,
    verbose=True
)

# 同一个链，自动路由到不同专家
print(router_chain.invoke("什么是量子纠缠？"))       # → physics
print(router_chain.invoke("如何用Python写快速排序？"))  # → coding

if __name__ == "__main__":
    print("OK")