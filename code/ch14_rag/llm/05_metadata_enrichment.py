# ---
# chapter: 14
# topic: 元数据增强（Metadata Enrichment）
# section: 14.3.5 元数据增强
# difficulty: ⭐⭐⭐
# tier: llm
# deps: (none)
# run: python 05_metadata_enrichment.py
# expected_runtime: <1s
# expected_output: structured chunk dict printed
# ---
# See: ../tutorial/14_RAG检索增强生成.md#14.3-文档处理与分块策略
# Interview hooks:
#   1. 给 chunk 添加元数据可以带来哪些检索收益？
#   2. 元数据过滤（filter by source/date）如何提升召回质量？
#   3. 实际生产中应该记录哪些元数据维度？

# 元数据增强示例
chunk_with_metadata = {
    "page_content": "年假政策：员工每年享有 15 天带薪年假...",
    "metadata": {
        "source": "公司人事手册_v2024.pdf",  # 来源文档
        "page": 15,                           # 页码
        "section": "第三章 休假制度",          # 章节
        "doc_type": "policy",                 # 文档类型
        "created_at": "2024-01-15",           # 创建日期
    },
}


if __name__ == "__main__":
    import json
    print(json.dumps(chunk_with_metadata, ensure_ascii=False, indent=2))
