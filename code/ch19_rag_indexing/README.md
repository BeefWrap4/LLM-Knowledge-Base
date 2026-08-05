# ch19_rag_indexing 代码伴侣

> 本目录是运行分组，不是主题身份。示例的规范归属以 [`code/TOPIC_MANIFEST.json`](../TOPIC_MANIFEST.json) 中的稳定 `topic_id` 和 `chapter` 为准。

## 对应章节

- [第 19 章 RAG 数据解析、分块与索引](../../19_RAG数据解析分块与索引.md)
- [第 20 章 RAG 检索、重排与高级方法](../../20_RAG检索重排与高级方法.md)
- [第 21 章 生产级 RAG 系统](../../21_生产级RAG系统.md)
- [第 37 章 RAG、Agent 与安全评估](../../37_RAG_Agent与安全评估.md)

## 示例统计

- 共 25 个示例：llm=25
- 每个示例保留 `# ---` metadata、`if __name__ == "__main__"` 入口和 `OK` 成功标记。

## 运行与验收

```powershell
python code/scripts/run_all_examples.py --chapter ch19 --tier core
python code/scripts/run_all_examples.py --chapter ch19 --tier llm
python code/scripts/run_all_examples.py --chapter ch19 --tier gpu
```

默认 LLM/GPU 验收使用 mock 或条件跳过；真实 API、GPU、模型下载和付费调用必须显式启用。
