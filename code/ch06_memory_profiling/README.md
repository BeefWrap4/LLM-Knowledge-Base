# ch06_memory_profiling 代码伴侣

> 本目录是运行分组，不是主题身份。示例的规范归属以 [`code/TOPIC_MANIFEST.json`](../TOPIC_MANIFEST.json) 中的稳定 `topic_id` 和 `chapter` 为准。

## 对应章节

- [第 6 章 Python 内存管理与性能诊断](../../06_Python内存管理与性能诊断.md)

## 示例统计

- 共 9 个示例：core=9
- 每个示例保留 `# ---` metadata、`if __name__ == "__main__"` 入口和 `OK` 成功标记。

## 运行与验收

```powershell
python code/scripts/run_all_examples.py --chapter ch06 --tier core
python code/scripts/run_all_examples.py --chapter ch06 --tier llm
python code/scripts/run_all_examples.py --chapter ch06 --tier gpu
```

默认 LLM/GPU 验收使用 mock 或条件跳过；真实 API、GPU、模型下载和付费调用必须显式启用。
