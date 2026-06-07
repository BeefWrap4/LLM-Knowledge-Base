# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.6.2 数据漂移检测（Embedding Drift）
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: numpy, scipy
# run: python 14_embedding_drift_detector.py
# expected_runtime: < 1s
# expected_output: drift detection dict printed (cosine distance + KS p-value)
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2062-数据漂移检测embedding-drift-⭐⭐
# Interview hooks:
#  - LLM 应用的"数据漂移"和传统特征漂移有什么本质区别？
#  - 余弦距离与 KS 检验在 Embedding 漂移检测中各自捕捉什么？
#  - 漂移阈值（0.1）是怎么拍出来的？能不能自适应？

from typing import Callable, Dict, List

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp


class EmbeddingDriftDetector:
    """基于 Embedding 的数据漂移检测器"""

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        reference_window_size: int = 1000,
        drift_threshold: float = 0.1,
    ):
        self.embed_fn = embed_fn
        self.window_size = reference_window_size
        self.threshold = drift_threshold
        self.reference_embeddings: List[np.ndarray] = []
        self.current_embeddings: List[np.ndarray] = []

    def add_reference(self, text: str):
        emb = np.array(self.embed_fn(text))
        self.reference_embeddings.append(emb)
        if len(self.reference_embeddings) > self.window_size:
            self.reference_embeddings.pop(0)

    def add_current(self, text: str):
        emb = np.array(self.embed_fn(text))
        self.current_embeddings.append(emb)
        if len(self.current_embeddings) > self.window_size:
            self.current_embeddings.pop(0)

    def detect_drift(self) -> Dict:
        if len(self.reference_embeddings) < 50 or len(self.current_embeddings) < 50:
            return {"drift_detected": False, "reason": "数据不足"}

        ref_array = np.array(self.reference_embeddings)
        cur_array = np.array(self.current_embeddings)

        ref_centroid = np.mean(ref_array, axis=0)
        cur_centroid = np.mean(cur_array, axis=0)
        centroid_distance = float(cosine(ref_centroid, cur_centroid))

        n_dims = ref_array.shape[1]
        ks_pvalues: List[float] = []
        for dim in range(min(n_dims, 10)):
            stat, pval = ks_2samp(ref_array[:, dim], cur_array[:, dim])
            ks_pvalues.append(float(pval))
        avg_ks_pvalue = float(np.mean(ks_pvalues))

        drift_detected = (
            centroid_distance > self.threshold
            or avg_ks_pvalue < 0.05
        )
        return {
            "drift_detected": drift_detected,
            "centroid_cosine_distance": round(centroid_distance, 4),
            "avg_ks_pvalue": round(avg_ks_pvalue, 4),
            "reference_samples": len(self.reference_embeddings),
            "current_samples": len(self.current_embeddings),
            "interpretation": (
                "⚠️ 检测到数据漂移！用户问题分布已发生变化，建议复查 Prompt 效果"
                if drift_detected
                else "✅ 数据分布稳定，未见明显漂移"
            ),
        }


if __name__ == "__main__":
    # 用一个稳定的随机 embed_fn 模拟：reference 与 current 来自不同分布
    rng_ref = np.random.default_rng(42)
    rng_cur = np.random.default_rng(99)

    def fake_embed(text: str) -> List[float]:
        # 模拟：current 文本的 embedding 整体平移以制造漂移
        base = rng_ref.normal(0, 1, size=16) if "ref" in text else rng_cur.normal(0.5, 1, size=16)
        return base.tolist()

    detector = EmbeddingDriftDetector(fake_embed)
    for i in range(120):
        detector.add_reference(f"ref text {i}")
    for i in range(120):
        detector.add_current(f"cur text {i}")
    print(detector.detect_drift())
