# 模型下载与使用指南

> 适用于 `code/scripts/download_models.py` 当前注册表。模型仓库、revision、许可、文件体积和
> 运行要求会变化；下载前以模型卡和 Hub dry-run 为准。

## 1. 先列清单

在 `code/` 目录执行：

```bash
python scripts/download_models.py --list
```

该命令只打印模型 ID、tier、默认状态、确认门禁和规划体积，不创建目录、不联网。
`required=True` 仅表示本教程的默认下载集合，不表示所有章节都必须安装。

## 2. 安全默认与按需下载

```bash
# 默认集合：embedding、reranker、0.5B 教学模型
python scripts/download_models.py --required-only

# 单项
python scripts/download_models.py --embedding
python scripts/download_models.py --reranker
python scripts/download_models.py --llm
python scripts/download_models.py --reasoner
python scripts/download_models.py --edge-mlx
python scripts/download_models.py --edge-gguf

# 大型或受许可集合：先核对模型卡、磁盘/显存和网络预算
python scripts/download_models.py --llm-medium --confirm-large
python scripts/download_models.py --world-model --confirm-large

# 全注册表；会包含大型和受许可模型
python scripts/download_models.py --all --confirm-large
```

Makefile 提供对应入口：

```bash
make download-models-list
make download-models-default
make download-models-llm
make download-models-gpu
make download-models-edge
```

下载脚本依次尝试 ModelScope、`hf-mirror.com` 和 Hugging Face 官方端点；需要精确文件过滤的
GGUF 仓库只走 Hugging Face 接口。镜像是否可用和实际速度取决于地区、时间、代理与缓存，
教程不承诺固定带宽或完成时间。

## 3. 当前注册模型的用途

| tier | 当前模型 | 主要章节 | 边界 |
|---|---|---|---|
| embedding | `BAAI/bge-small-zh-v1.5` | Ch14/17/20/22 | 默认集合 |
| reranker | `BAAI/bge-reranker-v2-m3` | Ch17/22 | 默认集合 |
| llm-small | `Qwen/Qwen2.5-0.5B-Instruct` | 教学加载/训练 | 默认集合；能力不代表生产模型 |
| llm-medium | Qwen2.5-7B、Llama 3.1 8B | Ch25 | 需确认；Llama 还可能要求 Hub 授权 |
| world-model/VLA | `nvidia/Cosmos3-Nano`、`lerobot/pi0_base` | Ch26 | 需确认；接口、硬件和许可分别核验 |
| reasoner | DeepSeek-R1-Distill-Qwen-1.5B | Ch27 | 蒸馏模型不等于云端 DeepSeek-R1 |
| edge | MLX Qwen 4-bit、单个 Llama GGUF | Ch28 | MLX 仅适合 Apple Silicon；GGUF 精确筛选 |

表中模型选择是教学注册表，不是质量排名或生产推荐。

## 4. 本地加载与验证

脚本下载到 `code/models/<local_name>/`。使用本地权重时应传明确路径，避免库在文件缺失时
自动联网：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/bge-small-zh-v1.5")
vectors = model.encode(["测试中文向量"])
print(vectors.shape)
```

Reranker 最小探针：

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("models/bge-reranker-v2-m3")
scores = model.predict([("什么是 RAG", "RAG 是检索增强生成")])
print(scores)
```

下载完成不等于章节验收通过。至少再核对：

1. 目标 sentinel/config 文件存在，Git LFS 文件不是指针；
2. 模型能从本地路径加载，进程未发起隐式下载；
3. dtype、量化格式、驱动/后端与目标硬件兼容；
4. 用章节的业务断言验证输出，而不只看进程退出码。

## 5. 许可、凭据与磁盘

- 先阅读当前模型卡、仓库许可、用途限制和 gated access 条件。
- Hugging Face token 只放环境变量或凭据存储，不写入仓库、日志或截图。
- `size_gb` 是规划值；多组件仓库、revision、量化和筛选规则都会改变实际体积。
- 不提交 `code/models/`、模型缓存或权重。
- 清理前先确认目录和硬链接关系，避免误删其他项目共用缓存。

## 6. 离线验收边界

整库默认验收不要求下载模型：

```bash
LLM_MOCK=1 python scripts/verify_all.py
python scripts/run_all_examples.py --tier gpu
```

GPU runner 默认传入 `--mock`，缺少真实模型/硬件的项目记为 `SKIP`，不能记为真实通过。
真实模型、GPU、网络和服务端路径需要在兼容环境单独验收并记录 revision、依赖、硬件和输出。

权威接口说明：

- [Hugging Face Hub 下载 API](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)
- [ModelScope SDK](https://github.com/modelscope/modelscope)
- [NVIDIA Cosmos](https://github.com/NVIDIA/cosmos)
- [LeRobot π0 模型卡](https://huggingface.co/lerobot/pi0_base)
