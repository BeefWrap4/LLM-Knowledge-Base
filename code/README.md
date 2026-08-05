# 可运行代码伴侣

> 54 章教程的 29 个代码运行分组：433 个示例（158 core + 199 llm + 76 gpu）

目录编号用于批量运行，不再等同于每个文件的永久主题归属。所有示例都有唯一 `topic_id`；[`TOPIC_MANIFEST.json`](TOPIC_MANIFEST.json) 记录其规范章节、路径和 tier。

## 快速验收

```powershell
python -m pip install -r requirements-core.txt
make ci-quick
make test
make lint
```

没有 API Key 时：

```powershell
$env:LLM_MOCK = "1"
make test-llm
```

运行一个分组或单文件：

```powershell
python scripts/run_all_examples.py --chapter ch22 --tier llm
python ch15_transformer/core/01_scaled_dot_product_attention.py
```

## 运行分组

| 目录 | 示例 | 规范章节 |
|---|---:|---|
| `ch01_python_runtime` | 22 | Ch01, Ch02, Ch03 |
| `ch02_object_model` | 18 | Ch02 |
| `ch04_iteration_functional` | 17 | Ch03, Ch04 |
| `ch05_oop_data_model` | 15 | Ch05 |
| `ch06_memory_profiling` | 9 | Ch06 |
| `ch07_concurrency` | 17 | Ch07 |
| `ch08_data_structures` | 11 | Ch08 |
| `ch09_numpy_pandas` | 13 | Ch09 |
| `ch10_fastapi` | 6 | Ch10 |
| `ch11_ml_basics` | 13 | Ch11 |
| `ch12_pytorch` | 11 | Ch12 |
| `ch15_transformer` | 6 | Ch14, Ch15, Ch31 |
| `ch17_prompt_engineering` | 22 | Ch17, Ch18, Ch28, Ch32, Ch38 |
| `ch18_context_engineering` | 12 | Ch18, Ch24, Ch26 |
| `ch19_rag_indexing` | 25 | Ch19, Ch20, Ch21, Ch37 |
| `ch22_agent_tools` | 22 | Ch22, Ch23, Ch24, Ch25, Ch26, Ch38 |
| `ch27_llm_frameworks` | 37 | Ch24, Ch26, Ch27, Ch30 |
| `ch29_data_engineering` | 14 | Ch29, Ch32 |
| `ch30_lora_qlora` | 15 | Ch30, Ch31, Ch40, Ch41, Ch46 |
| `ch32_reasoning_ttc` | 14 | Ch32 |
| `ch33_distributed` | 11 | Ch33 |
| `ch36_evaluation` | 14 | Ch36, Ch37 |
| `ch38_safety` | 14 | Ch24, Ch38, Ch39 |
| `ch41_inference_engines` | 12 | Ch40, Ch41, Ch42 |
| `ch43_cloudnative` | 7 | Ch43, Ch44, Ch46 |
| `ch44_llmops` | 25 | Ch44, Ch45 |
| `ch46_edge_llm` | 10 | Ch46 |
| `ch47_multimodal` | 11 | Ch21, Ch47, Ch48, Ch49 |
| `ch49_world_models` | 10 | Ch49 |

## 示例契约

- Python 3.10+，四空格，Ruff 行宽 110。
- 文件开头包含 `# ---` metadata、`chapter`、稳定 `topic_id`、tier、依赖和预期结果。
- 脚本可直接运行，保留 `if __name__ == "__main__"`，成功输出 `OK`。
- 缺少可选依赖、真实 API 或 GPU 时输出明确 `[SKIP]`，不得伪造成功。
- 模型权重、缓存、`.env` 和 API Key 不得提交。

## 主题覆盖

当前 433 个示例覆盖 45/54 个规范章节。没有独立示例的专题章通过设计题、跨章示例或相邻运行分组验收，不为凑覆盖率复制代码。
