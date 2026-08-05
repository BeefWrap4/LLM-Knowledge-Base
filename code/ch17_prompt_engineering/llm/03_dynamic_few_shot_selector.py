# ---
# chapter: 17
# topic: Prompt Engineering
# topic_id: prompt_engineering.dynamic_few_shot_selector
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: sentence-transformers (可选, 缺失时使用 mock 嵌入), numpy
# run: python 03_dynamic_few_shot_selector.py
# expected_runtime: 2-10s (依赖模型下载)
# expected_output: 打印检索到的与查询最相似的示例
# ---
# See: ../../../17_Prompt_Engineering.md
# Interview hooks:
# - 动态 Few-shot 比固定 Few-shot 好在哪里？
# - 为何使用归一化嵌入 + 点积等价于余弦相似度？
# - 当候选示例规模达到百万级时，如何高效检索？(ANN/Faiss)


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
except Exception:
    HAS_ST = False


class _MockEmbedder:
    """缺失 sentence_transformers 时使用的哈希启发式嵌入。"""

    def encode(self, texts, normalize_embeddings: bool = True):
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        rng = np.random.RandomState(42)
        base = rng.randn(len(texts), 64).astype(np.float32)
        # 同样的文本得到同样的向量（基于 hash）
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2**32 - 1)
            r = np.random.RandomState(seed)
            base[i] = r.randn(64).astype(np.float32)
        if normalize_embeddings:
            base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-9
        return base[0] if single else base


class DynamicFewShotSelector:
    """动态 Few-shot 示例选择器：基于语义相似度检索最相关的示例"""

    def __init__(self, examples: list[dict]):
        """
        Args:
            examples: [{"input": str, "output": str, "task_type": str}, ...]
        """
        self.examples = examples
        if HAS_ST:
            try:
                self.embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            except Exception as e:
                # 网络/HF 离线时 fallback 到 mock
                print(f"[mock] SentenceTransformer 加载失败 ({type(e).__name__}), 使用哈希嵌入")
                self.embedder = _MockEmbedder()
        else:
            print("[mock] sentence-transformers 未安装，使用哈希嵌入演示")
            self.embedder = _MockEmbedder()
        self.embeddings = self.embedder.encode([ex["input"] for ex in examples], normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """检索与 query 最相似的 top_k 个示例"""
        query_vec = self.embedder.encode(query, normalize_embeddings=True)
        # 余弦相似度 = 归一化后的点积
        similarities = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.examples[i] for i in top_indices]


if __name__ == "__main__":
    # 使用示例
    examples_db = [
        {"input": "这电影真好看", "output": "正面", "task_type": "情感分析"},
        {"input": "服务态度太差", "output": "负面", "task_type": "情感分析"},
        {"input": "一般般吧", "output": "中性", "task_type": "情感分析"},
        {"input": "物流配送非常迅速", "output": "正面", "task_type": "情感分析"},
        {"input": "客服解决问题很专业", "output": "正面", "task_type": "情感分析"},
    ]

    selector = DynamicFewShotSelector(examples_db)
    relevant_examples = selector.retrieve("物流速度很快", top_k=2)
    print("[查询] 物流速度很快")
    print("[Top-2 相似示例]")
    for ex in relevant_examples:
        print(f"  - {ex['input']} → {ex['output']}")
    print("OK")
