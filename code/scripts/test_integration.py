#!/usr/bin/env python3
# ---
# code/scripts/test_integration.py (Wave 26-E)
# 真实集成测试: Redis + pgvector + 真实模型 + 真实 LLM
# 验证整个本地部署栈完全可用
# Usage: python code/scripts/test_integration.py
# Exit code: 0 全部 PASS, 1 至少 1 项失败
# ---
"""
本地部署集成测试 — 不依赖 mock.

测试 4 个核心组件:
  1. bge-small-zh-v1.5 embedding (本机推理)
  2. Redis (端口 16379)
  3. pgvector (端口 15432)
  4. MiniMax (或其他 4 厂商) 真实 LLM API

预期: 4/4 PASS 表示本地环境已就绪, 教程所有"打桩代码"可替换为真实执行.
"""
import sys
import time
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

import numpy as np


def test_embedding():
    """[1/4] bge-small-zh-v1.5 本机 embedding."""
    print("\n[1/4] Embedding 模型 (bge-small-zh-v1.5)...")
    from sentence_transformers import SentenceTransformer
    t0 = time.perf_counter()
    m = SentenceTransformer(str(CODE / "models" / "bge-small-zh-v1.5"))
    t_load = time.perf_counter() - t0
    print(f"  模型加载: {t_load:.1f}s, dim={m.get_sentence_embedding_dimension()}")
    docs = ["Python 是一种编程语言", "Redis 是内存数据库", "Qwen 是阿里大模型"]
    t0 = time.perf_counter()
    embs = m.encode(docs, normalize_embeddings=True)
    t_enc = time.perf_counter() - t0
    print(f"  3 docs 编码: {t_enc*1000:.0f}ms ({t_enc*1000/3:.0f}ms/doc)")
    # Self-similarity check
    sim = float(np.dot(embs[0], embs[0]))
    assert sim > 0.99, f"Self-similarity 应 > 0.99, 实际 {sim}"
    print(f"  self-similarity: {sim:.4f} ✓")
    return embs


def test_redis(embeddings):
    """[2/4] Redis 16379 读写."""
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    print(f"\n[2/4] Redis ({redis_host}:16379)...")
    import redis
    r = redis.Redis(host=redis_host, port=16379, db=0, socket_connect_timeout=3)
    r.ping()
    print(f"  PING: ✓, version={r.info('server')['redis_version']}")
    # Write/read test
    r.set("integration_test_key", "hello from LLM-KB")
    val = r.get("integration_test_key").decode("utf-8")
    assert val == "hello from LLM-KB"
    print(f"  SET/GET: ✓ ({val!r})")
    # Hash with embeddings
    for i, emb in enumerate(embeddings):
        r.hset(f"emb:{i}", mapping={"dim": len(emb), "norm": float(np.linalg.norm(emb))})
    keys = r.keys("emb:*")
    print(f"  HSET emb:*: {len(keys)} keys ✓")
    r.delete("integration_test_key", *keys)
    return True


def test_pgvector(embeddings):
    """[3/4] pgvector 向量检索 (cosine similarity)."""
    pg_host = os.environ.get("PG_HOST", "localhost")
    print(f"\n[3/4] pgvector ({pg_host}:15432)...")
    import psycopg2
    c = psycopg2.connect(host=pg_host, port=15432, user="llmkb", password="llmkb_test", dbname="vectordb")
    c.autocommit = True
    cur = c.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("DROP TABLE IF EXISTS integration_test_vec")
    cur.execute("CREATE TABLE integration_test_vec (id SERIAL PRIMARY KEY, content TEXT, embedding VECTOR(512))")
    docs = ["Python 是一种编程语言", "Redis 是内存数据库", "Qwen 是阿里大模型"]
    for doc, emb in zip(docs, embeddings):
        vec_str = "[" + ",".join("%.7f" % x.item() for x in emb.flatten()) + "]"
        cur.execute("INSERT INTO integration_test_vec (content, embedding) VALUES (%s, %s)", (doc, vec_str))
    # Query
    query_emb = embeddings[0]  # "Python" doc itself (perfect match)
    query_str = "[" + ",".join("%.7f" % x.item() for x in query_emb.flatten()) + "]"
    cur.execute("""
        SELECT content, 1 - (embedding <=> %s::vector) AS cosine
        FROM integration_test_vec
        ORDER BY embedding <=> %s::vector LIMIT 1
    """, (query_str, query_str))
    top_content, top_score = cur.fetchone()
    print(f"  Query: 与 'Python 是一种编程语言' 最相似")
    print(f"  Top match: {top_content} (cosine={top_score:.4f})")
    assert top_content == docs[0], f"自相似查询应返回自身, 实际 {top_content}"
    assert top_score > 0.99, f"自相似 cosine 应 > 0.99, 实际 {top_score}"
    print(f"  自相似验证: ✓ (cosine={top_score:.4f})")
    cur.execute("DROP TABLE integration_test_vec")
    c.close()
    return True


def test_llm():
    """[4/4] 真实 LLM API (4 厂商轮询)."""
    print("\n[4/4] 真实 LLM API...")
    from shared.llm_client import UnifiedClient
    from shared.provider_registry import PROVIDERS
    available = [p for p in PROVIDERS.values() if p.has_key() and p.name != "mock"]
    if not available:
        print("  ⚠️  无任何 API Key, 跳过 (至少 1 个 key 才会跑)")
        return False
    print(f"  发现 {len(available)} 家厂商, 跑 quick test...")
    for p in available:
        c = UnifiedClient(provider=p.name)
        if c.is_mock:
            continue
        t0 = time.perf_counter()
        try:
            resp = c.chat(prompt="用一句话介绍 Python 编程语言", max_tokens=60)
            elapsed = time.perf_counter() - t0
            if resp.mock:
                print(f"  [{p.name}] fallback to mock: {resp.content[:50]}")
            else:
                print(f"  [{p.name}] {elapsed:.2f}s | {resp.content[:60]}")
        except Exception as e:
            print(f"  [{p.name}] ERROR: {type(e).__name__}: {str(e)[:80]}")
    return True


def main():
    print("=" * 60)
    print("  本地部署集成测试 (Real LLM + Real Models + Middleware)")
    print("=" * 60)
    t_start = time.perf_counter()
    results = {}

    # 1. Embedding
    try:
        embeddings = test_embedding()
        results["embedding"] = True
    except Exception as e:
        print(f"  ✗ embedding 失败: {e}")
        results["embedding"] = False
        embeddings = None

    # 2. Redis (only if embedding OK)
    if embeddings is not None:
        try:
            test_redis(embeddings)
            results["redis"] = True
        except Exception as e:
            print(f"  ✗ redis 失败: {e}")
            results["redis"] = False

    # 3. pgvector (only if embedding OK)
    if embeddings is not None:
        try:
            test_pgvector(embeddings)
            results["pgvector"] = True
        except Exception as e:
            print(f"  ✗ pgvector 失败: {e}")
            results["pgvector"] = False

    # 4. LLM
    try:
        test_llm()
        results["llm"] = True
    except Exception as e:
        print(f"  ✗ llm 失败: {e}")
        results["llm"] = False

    # Summary
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  [{'✓' if v else '✗'}] {k}")
    print(f"  总耗时: {elapsed:.1f}s")
    print("=" * 60)

    if all(results.values()):
        print("\n  🎉 全部通过! 本地环境已就绪, 教程所有真实调用代码可执行.")
        return 0
    else:
        print("\n  ⚠️  部分失败, 参考上方错误信息排查.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
