# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.5.3 长期记忆：向量存储 + 知识图谱
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: [numpy]  # sentence-transformers 是可选依赖，未安装时自动回退
# run: python 07_long_term_memory.py
# expected_runtime: <1s（无 sentence-transformers 时为纯 numpy 演示）
# expected_output: 相似度排序的 top-k 经验 + 知识图谱查询
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.5.3-长期记忆向量存储-知识图谱
# Interview hooks:
#   1. 长期记忆为什么一般用向量数据库而不是直接保存字符串？
#   2. 经验记忆和事实记忆的存储结构为何不同？分别适用于哪些查询？
#   3. 当 sentence-transformers 加载很慢时，工程上如何兜底？(降级到 hash 特征)



# === Optional dependency guard (auto-added) ===
import sys as _sys
try:
    from sentence_transformers import SentenceTransformer
    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    _sys.exit(0)
import numpy as np

try:
    HAS_ST = True
except Exception:  # pragma: no cover
    HAS_ST = False


def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
    """sentence-transformers 不可用时的轻量回退：基于字符 n-gram 的哈希向量"""
    vec = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(text):
        vec[hash(ch) % dim] += 1
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


class LongTermMemory:
    """
    Agent 长期记忆系统

    包含两个存储：
    1. 经验记忆（向量数据库）：存储历史对话和经验
    2. 事实记忆（知识图谱）：存储实体关系
    """

    def __init__(self, embedding_model: str = "BAAI/bge-small-zh-v1.5"):
        if HAS_ST:
            try:
                self.embedder = SentenceTransformer(embedding_model)
            except Exception as e:
                # 网络/HF 离线时 fallback 到 None (下游代码有 None 检查)
                print(f"[mock] SentenceTransformer 加载失败 ({type(e).__name__})，降级到 hash 特征")
                self.embedder = None
        else:
            self.embedder = None

        # 经验记忆: [(text, embedding, metadata), ...]
        self.experiences: list[dict] = []

        # 事实记忆: {entity: {relation: target, ...}, ...}
        self.facts: dict[str, dict[str, str]] = {}

    def _encode(self, text: str) -> np.ndarray:
        if self.embedder is not None:
            return self.embedder.encode(text, normalize_embeddings=True)
        return _hash_embed(text)

    def add_experience(self, text: str, experience_type: str = "conversation"):
        """添加经验记忆"""
        embedding = self._encode(text)
        self.experiences.append({
            "text": text,
            "embedding": embedding,
            "metadata": {"type": experience_type, "timestamp": "now"}
        })

    def retrieve_relevant(self, query: str, top_k: int = 3) -> list[str]:
        """检索相关经验"""
        if not self.experiences:
            return []

        query_emb = self._encode(query)

        # 计算余弦相似度
        experiences_array = np.array([e["embedding"] for e in self.experiences])
        similarities = np.dot(experiences_array, query_emb)

        # 取 Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.experiences[i]["text"] for i in top_indices]

    def add_fact(self, entity: str, relation: str, target: str):
        """添加事实记忆"""
        if entity not in self.facts:
            self.facts[entity] = {}
        self.facts[entity][relation] = target

    def query_fact(self, entity: str, relation: str = None):
        """查询事实"""
        if entity not in self.facts:
            return {}
        if relation:
            return self.facts[entity].get(relation, "未知")
        return self.facts[entity]


# 完整记忆管理集成
class AgentMemory:
    """Agent 完整记忆系统"""

    def __init__(self, stm_max_tokens: int = 4000):
        from collections import deque
        self._stm_messages: deque = deque()
        self.ltm = LongTermMemory()
        self.working_memory: dict = {}

    def memorize_interaction(self, user_msg: str, assistant_msg: str):
        """记录一次交互到短期记忆和长期记忆"""
        self._stm_messages.append({"role": "user", "content": user_msg})
        self._stm_messages.append({"role": "assistant", "content": assistant_msg})
        # 简化：直接将拼接文本存入 LTM
        self.ltm.add_experience(f"User: {user_msg}\nAssistant: {assistant_msg}")

    def get_context(self, current_query: str) -> list[dict]:
        """
        获取完整上下文：
        1. 短期记忆（对话历史）
        2. 从长期记忆中检索相关经验
        """
        messages = list(self._stm_messages)

        # 检索长期记忆中的相关经验
        relevant_experiences = self.ltm.retrieve_relevant(current_query, top_k=2)

        if relevant_experiences:
            # 将相关经验作为上下文注入
            context_msg = "相关历史经验：\n" + "\n---\n".join(relevant_experiences)
            messages.insert(0, {"role": "system", "content": context_msg})

        return messages


def main():
    ltm = LongTermMemory()
    ltm.add_experience("用户问过 Python GIL，答案是 CPython 互斥锁机制")
    ltm.add_experience("用户问过 LoRA，用于参数高效微调")
    ltm.add_experience("用户问过 RAG，检索增强生成")

    top = ltm.retrieve_relevant("请解释 RAG 的核心思想", top_k=2)
    print("=== Top-2 相关经验 ===")
    for t in top:
        print(f"  - {t}")

    # 知识图谱
    ltm.add_fact("北京", "位于", "中国")
    ltm.add_fact("北京", "人口", "约 2200 万")
    print(f"\n北京人口: {ltm.query_fact('北京', '人口')}")
    print(f"北京所有事实: {ltm.query_fact('北京')}")

    print("\nOK")


if __name__ == "__main__":
    main()