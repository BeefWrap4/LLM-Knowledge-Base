# ---
# chapter: 17
# topic: 大模型评估体系
# section: 17.5.2 Ragas 框架
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: ragas, datasets, langchain-openai, langchain-huggingface
# run: python 06_ragas_evaluation.py
# expected_runtime: <5s (mock mode) / 30-60s (real)
# expected_output: RAG evaluation scores for faithfulness, relevancy, recall, precision
# ---
# See: ../tutorial/17_大模型评估体系.md
# Interview hooks:
# - Explain the difference between Faithfulness and Answer Relevancy.
# - How does Ragas compute Context Recall without ground-truth labels?
# - What are the limitations of LLM-as-Judge based RAG metrics?

"""Ragas 实战评估示例 (Wave 30: 真实 LLM + 本地 bge embedding).

Ragas（RAG Assessment）是最流行的开源 RAG 评估框架，
提供忠实度、答案相关性、上下文召回率、上下文精确率等核心指标。

Wave 30 改进:
  - LLM 改用 UnifiedClient (MiniMax / DeepSeek / Kimi / SiliconFlow 任意厂商)
  - Embedding 改用本地 bge-small-zh-v1.5 (无需 OpenAI Key)
"""
import sys as _sys_path_setup
from pathlib import Path as _Path_setup
_code_root = _Path_setup(__file__).resolve().parent.parent.parent
if str(_code_root) not in _sys_path_setup.path:
    _sys_path_setup.path.insert(0, str(_code_root))

import os


def run_ragas_evaluation() -> None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            answer_correctness,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        print(f"[mock] 依赖未安装 ({exc}), 使用模拟输出")
        print("  pip install ragas datasets langchain-huggingface")
        return

    # 准备评估数据
    eval_dataset = Dataset.from_dict(
        {
            "question": [
                "什么是Transformer的注意力机制？",
                "Python中的GIL是什么？",
            ],
            "answer": [
                "注意力机制是Transformer的核心组件, 它允许模型在处理序列时动态关注不同位置的信息。"
                "通过计算Query、Key、Value之间的相似度, 模型可以为不同位置的token分配不同的权重, "
                "从而捕捉长距离依赖关系。",
                "GIL(全局解释器锁)是CPython中的一种互斥锁, 它确保同一时刻只有一个线程执行Python字节码。"
                "这简化了CPython的内存管理, 但也限制了多线程程序的并行性能。",
            ],
            "contexts": [
                [
                    "注意力机制通过Q、K、V三个矩阵计算token间的关系。",
                    "Transformer架构由Vaswani等人在2017年提出, 核心创新是自注意力机制。",
                    "自注意力可以捕捉序列中任意两个位置之间的依赖关系。",
                ],
                [
                    "GIL是CPython解释器中的全局锁, 用于保护内部数据结构。",
                    "由于GIL的存在, 多线程Python程序在CPU密集型任务中无法充分利用多核。",
                    "可以通过multiprocessing模块或使用PyPy/Jython等替代解释器绕过GIL。",
                ],
            ],
            "ground_truth": [
                "注意力机制是Transformer中让模型关注输入序列中不同部分重要性的机制, 通过Q、K、V计算实现。",
                "GIL是CPython的全局解释器锁, 确保线程安全但限制了多线程并行。",
            ],
        }
    )

    # Wave 30: LLM 用 chatmodel_factory (支持 7 厂商)
    from shared.chatmodel_factory import make_chat_model
    lc_llm = make_chat_model(provider=os.environ.get("LLM_PROVIDER", "MiniMax"))
    if lc_llm is None:
        print("[mock] UnifiedClient 无 Key, 使用模拟输出")
        print("  设 DEEPSEEK_API_KEY 等后重试")
        return
    llm = LangchainLLMWrapper(lc_llm)
    print(f"[ragas] LLM: {lc_llm.model_name if hasattr(lc_llm, 'model_name') else '?'}")

    # Wave 30: Embedding 用本地 bge-small-zh-v1.5 (无需 OpenAI)
    bge_path = _code_root / "models" / "bge-small-zh-v1.5"
    if not (bge_path.exists() and (bge_path / "config.json").exists()):
        print(f"[WARN] 本地 bge 不存在: {bge_path}")
        print("  跑 setup_local.sh 下载 bge-small-zh-v1.5 模型")
        return
    emb = HuggingFaceEmbeddings(model_name=str(bge_path))
    embeddings = LangchainEmbeddingsWrapper(emb)
    print(f"[ragas] Embedding: bge-small-zh (本地)")

    # 选择评估指标
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    ]

    # 执行评估 (faithfulness 需要 LLM, answer_relevancy 需要 embedding)
    print("[ragas] 跑评估 (~30-60s)...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    # 输出结果
    df = result.to_pandas()
    print("\nRAG 评估结果:")
    cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].to_string(index=False))


if __name__ == "__main__":
    run_ragas_evaluation()
    print("OK")
