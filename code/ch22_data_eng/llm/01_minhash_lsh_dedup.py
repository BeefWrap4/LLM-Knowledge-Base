# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.2.2 去重技术 - MinHash LSH 去重
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: datasketch
# run: python 01_minhash_lsh_dedup.py
# expected_runtime: <5s
# expected_output: 与 doc_1 相似的文档列表
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. MinHash LSH 的核心数学原理是什么？P[minhash(A)=minhash(B)] = J(A,B) 如何推导？
#   2. 为什么三层去重架构（精确 → MinHash LSH → SimHash）优于单层方案？
#   3. LSH 是如何将 O(n²) 的相似文档对比较降低到近似 O(n) 的？


# === Optional dependency guard (auto-added) ===
import sys as _sys

try:
    from datasketch import MinHash, MinHashLSH

    _SKIP_REASON = None
except (ImportError, ModuleNotFoundError) as _e:
    _SKIP_REASON = str(_e).split("\n")[0]
if _SKIP_REASON:
    print(f"[SKIP] {__file__}: {_SKIP_REASON}")
    _sys.exit(0)
import re


def tokenize(text: str, n: int = 3) -> list[str]:
    """将文本转为 n-gram 字符集合"""
    text = re.sub(r"\s+", " ", text.lower())
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    """为文档创建 MinHash 签名"""
    m = MinHash(num_perm=num_perm)
    for shingle in tokenize(text, n=5):
        m.update(shingle.encode("utf-8"))
    return m


def main():
    # 创建 LSH 索引
    lsh = MinHashLSH(threshold=0.8, num_perm=128)

    documents = [
        ("doc_1", "The quick brown fox jumps over the lazy dog"),
        ("doc_2", "The quick brown fox jumps over the lazy dog."),  # 几乎相同
        ("doc_3", "Machine learning is transforming data engineering"),  # 完全不同
        ("doc_4", "A quick brown fox leaped over a lazy dog"),  # 相似但不相同
    ]

    for doc_id, text in documents:
        mh = create_minhash(text)
        lsh.insert(doc_id, mh)

    # 查询重复文档
    target_mh = create_minhash(documents[0][1])
    duplicates = lsh.query(target_mh)
    print(f"与 doc_1 相似的文档（Jaccard >= 0.8）: {duplicates}")
    # 预期输出: ['doc_1', 'doc_2', 'doc_4']


if __name__ == "__main__":
    main()
    print("OK")
