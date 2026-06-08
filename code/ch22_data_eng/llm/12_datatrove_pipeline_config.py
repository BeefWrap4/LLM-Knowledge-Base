# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.6.2 Datatrove 数据处理 Pipeline
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: stdlib only
# run: python 12_datatrove_pipeline_config.py
# expected_runtime: <1s
# expected_output: Pipeline 配置字典的统计概览
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. Datatrove 的 Reader → Processing Blocks → Writer 流水线架构设计原则是什么？
#   2. "Gopher Quality Filter" 使用的困惑度阈值是多少？为什么选择 1000？
#   3. 多级 Pipeline 中如何处理节点失败和重试，保证 TB-PB 级数据处理的可靠性？

import json

# Datatrove 的 Pipeline 概念:
# 1. Reader: 读取原始数据
# 2. Processing Blocks: 一系列处理步骤
# 3. Writer: 写入处理后的数据

# Pipeline 配置示例（JSON格式概念）
datatrove_pipeline_config = {
    "reader": {
        "type": "JsonlReader",
        "data_folder": "/data/raw_cc/",
        "glob_pattern": "*.jsonl.gz",
        "text_key": "text",
        "id_key": "doc_id",
    },
    "processing": [
        {
            "type": "GopherQualityFilter",  # 基于困惑度的质量过滤
            "model_name": "gpt2",
            "threshold": 1000,
        },
        {
            "type": "MinHashDedup",  # MinHash 去重
            "num_perm": 128,
            "threshold": 0.8,
            "ngram_size": 5,
        },
        {
            "type": "LanguageFilter",  # 语言过滤
            "language": "en",
            "confidence_threshold": 0.65,
        },
        {
            "type": "PIIRemover",  # PII 去除
            "email": True,
            "phone": True,
            "ip": True,
        },
    ],
    "writer": {
        "type": "JsonlWriter",
        "output_folder": "/data/cleaned_corpus/",
        "compression": "gzip",
    },
}


def main():
    print("=== Datatrove Pipeline 配置概览 ===")
    print(f"Reader: {datatrove_pipeline_config['reader']['type']}")
    print(f"  数据源: {datatrove_pipeline_config['reader']['data_folder']}")
    print(f"\n处理步骤数: {len(datatrove_pipeline_config['processing'])}")
    for i, step in enumerate(datatrove_pipeline_config["processing"], 1):
        print(
            f"  [{i}] {step['type']}: {json.dumps({k: v for k, v in step.items() if k != 'type'}, ensure_ascii=False)}"
        )
    print(f"\nWriter: {datatrove_pipeline_config['writer']['type']}")
    print(f"  输出: {datatrove_pipeline_config['writer']['output_folder']}")

    # 实际运行命令（注释提示）
    print("\n# 实际运行命令（多机分布式）:")
    print("# python -m datatrove.run_pipeline \\")
    print("#   --reader.jsonl_reader.path /data/raw_cc/ \\")
    print("#   --writer.jsonl_writer.path /data/cleaned_corpus/ \\")
    print("#   --processing_blocks gopher_quality_filter.minhash_dedup \\")
    print("#   --num_proc 64 --main_tasks_per_node 8")


if __name__ == "__main__":
    main()
