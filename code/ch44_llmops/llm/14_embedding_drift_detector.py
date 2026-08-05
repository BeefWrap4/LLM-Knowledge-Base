# ---
# chapter: 45
# topic: 大模型可观测性与 SRE
# topic_id: llmops.embedding_drift_detector
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: numpy, scipy
# run: python 14_embedding_drift_detector.py
# expected_runtime: < 1s
# expected_output: drift detection dict printed (cosine distance + KS p-value)
# ---
# See: ../../../45_大模型可观测性与SRE.md
# Interview hooks:
#  - LLM 应用的"数据漂移"和传统特征漂移有什么本质区别？
#  - 余弦距离与 KS 检验在 Embedding 漂移检测中各自捕捉什么？
#  - 质心距离与多重 KS 检验阈值应如何用基线窗口和下游质量校准？

from collections.abc import Callable

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp


class EmbeddingDriftDetector:
    """基于 Embedding 的数据漂移检测器"""

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        *,
        centroid_distance_threshold: float,
        ks_familywise_alpha: float,
        reference_window_size: int = 1000,
        min_samples_per_window: int = 50,
        max_ks_dimensions: int = 10,
    ):
        if not 0 <= centroid_distance_threshold <= 2:
            raise ValueError("centroid_distance_threshold must be in [0, 2]")
        if not 0 < ks_familywise_alpha < 1:
            raise ValueError("ks_familywise_alpha must be in (0, 1)")
        if reference_window_size < min_samples_per_window or min_samples_per_window < 2:
            raise ValueError("window size must be >= min_samples_per_window >= 2")
        if max_ks_dimensions <= 0:
            raise ValueError("max_ks_dimensions must be positive")
        self.embed_fn = embed_fn
        self.window_size = reference_window_size
        self.centroid_threshold = centroid_distance_threshold
        self.ks_familywise_alpha = ks_familywise_alpha
        self.min_samples = min_samples_per_window
        self.max_ks_dimensions = max_ks_dimensions
        self.reference_embeddings: list[np.ndarray] = []
        self.current_embeddings: list[np.ndarray] = []

    def _embed(self, text: str) -> np.ndarray:
        embedding = np.asarray(self.embed_fn(text), dtype=float)
        if embedding.ndim != 1 or embedding.size == 0 or not np.isfinite(embedding).all():
            raise ValueError("embedding must be a non-empty, finite 1-D vector")
        expected_dim = next(
            (
                item.size
                for collection in (self.reference_embeddings, self.current_embeddings)
                for item in collection
            ),
            embedding.size,
        )
        if embedding.size != expected_dim:
            raise ValueError(f"embedding dimension changed: expected {expected_dim}, got {embedding.size}")
        return embedding

    def add_reference(self, text: str):
        self.reference_embeddings.append(self._embed(text))
        if len(self.reference_embeddings) > self.window_size:
            self.reference_embeddings.pop(0)

    def add_current(self, text: str):
        self.current_embeddings.append(self._embed(text))
        if len(self.current_embeddings) > self.window_size:
            self.current_embeddings.pop(0)

    def detect_drift(self) -> dict:
        if (
            len(self.reference_embeddings) < self.min_samples
            or len(self.current_embeddings) < self.min_samples
        ):
            return {
                "drift_detected": None,
                "status": "insufficient_data",
                "required_per_window": self.min_samples,
            }

        ref_array = np.array(self.reference_embeddings)
        cur_array = np.array(self.current_embeddings)

        ref_centroid = np.mean(ref_array, axis=0)
        cur_centroid = np.mean(cur_array, axis=0)
        if np.linalg.norm(ref_centroid) == 0 or np.linalg.norm(cur_centroid) == 0:
            centroid_distance = 0.0 if np.allclose(ref_centroid, cur_centroid) else 1.0
        else:
            centroid_distance = float(cosine(ref_centroid, cur_centroid))

        n_dims = ref_array.shape[1]
        ks_pvalues: list[float] = []
        tested_dimensions = min(n_dims, self.max_ks_dimensions)
        for dim in range(tested_dimensions):
            _statistic, pval = ks_2samp(ref_array[:, dim], cur_array[:, dim])
            ks_pvalues.append(float(pval))
        # 对逐维 KS 做 Bonferroni 控制；平均 p-value 不是有效的显著性检验。
        per_dimension_alpha = self.ks_familywise_alpha / tested_dimensions
        minimum_ks_pvalue = min(ks_pvalues)
        ks_drift_detected = minimum_ks_pvalue < per_dimension_alpha

        drift_detected = (
            centroid_distance > self.centroid_threshold or ks_drift_detected
        )
        return {
            "drift_detected": drift_detected,
            "status": "evaluated",
            "centroid_cosine_distance": round(centroid_distance, 4),
            "centroid_distance_threshold": self.centroid_threshold,
            "minimum_ks_pvalue": round(minimum_ks_pvalue, 6),
            "ks_per_dimension_alpha": round(per_dimension_alpha, 6),
            "ks_dimensions_tested": tested_dimensions,
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

    def fake_embed(text: str) -> list[float]:
        # 模拟：current 文本的 embedding 整体平移以制造漂移
        base = rng_ref.normal(0, 1, size=16) if "ref" in text else rng_cur.normal(0.5, 1, size=16)
        return base.tolist()

    # 教学阈值只为演示；生产中应以 reference-vs-reference 回放和下游质量变化校准。
    detector = EmbeddingDriftDetector(
        fake_embed,
        centroid_distance_threshold=0.1,
        ks_familywise_alpha=0.05,
    )
    for i in range(120):
        detector.add_reference(f"ref text {i}")
    for i in range(120):
        detector.add_current(f"cur text {i}")
    print(detector.detect_drift())
    print("OK")
