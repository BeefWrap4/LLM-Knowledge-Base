# 模型下载目录 (本地权重)

> 本目录存放教程代码中使用的真实模型权重, **不通过 git 跟踪** (`.gitignore` 已忽略).
> 推荐使用 **国内源** (ModelScope / HF 镜像) 下载, 速度比直连 HF 快 5-10 倍.

## 目录结构

```
code/models/
├── README.md                      # 本文件
├── bge-small-zh-v1.5/             # Embedding (100MB) — Ch14 RAG
├── bge-reranker-v2-m3/            # Reranker (600MB) — Ch14 高级 RAG
└── Qwen2.5-0.5B-Instruct/         # 小型 LLM (1GB) — Ch27 reasoning mock
```

## 一键下载

```bash
cd code/

# 默认: 仅下载轻量模型 (bge-small-zh + bge-reranker, ~700MB, 5-10 min)
make download-models

# 全部 (含 Qwen 0.5B, ~1.7GB, 20-30 min)
make download-models ARGS="--all"

# 仅 embedding
make download-models ARGS="--embedding"

# 仅 reranker
make download-models ARGS="--reranker"

# 仅 LLM
make download-models ARGS="--llm"
```

## 手动下载 (国内源)

### 1. ModelScope (推荐, 国内 CDN)

```bash
# 安装 ModelScope CLI
pip install modelscope

# 下载 bge-small-zh-v1.5 (embedding)
modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir code/models/bge-small-zh-v1.5

# 下载 bge-reranker-v2-m3
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir code/models/bge-reranker-v2-m3

# 下载 Qwen2.5-0.5B-Instruct
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir code/models/Qwen2.5-0.5B-Instruct
```

### 2. HuggingFace 镜像 (hf-mirror.com)

```bash
# 安装 HF CLI
pip install -U huggingface_hub

# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载 (路径与 ModelScope 一致)
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir code/models/bge-small-zh-v1.5
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir code/models/bge-reranker-v2-m3
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir code/models/Qwen2.5-0.5B-Instruct
```

### 3. 直连 HuggingFace (海外)

仅推荐海外用户. 国内极慢, 经常 timeout:

```bash
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir code/models/bge-small-zh-v1.5
```

## 选型指南

| 教程章节 | 推荐模型 | 用途 | 必需? |
|---------|---------|------|-------|
| Ch14 RAG | `bge-small-zh-v1.5` | 中文句向量, 100MB | 推荐 |
| Ch14 高级 RAG | `bge-reranker-v2-m3` | 二次排序 | 可选 |
| Ch22 数据去重 | `bge-small-zh-v1.5` | 语义指纹 | 推荐 |
| Ch16/27 推理 mock | `Qwen2.5-0.5B` | 极小 LLM | 可选 (大多数例子用 mock) |
| Ch28 端侧 | 任何 GGUF 量化 | 笔记本推理 | 可选 (见 Ch28 README) |

**教程价值最大化**: 下载 `bge-small-zh-v1.5` 就足以跑通 80% RAG / 语义搜索例子. 其他按需.

## 验证下载

下载完成后, 验证模型可加载:

```bash
cd code/

# 验证 embedding
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('models/bge-small-zh-v1.5')
v = m.encode(['测试句子'])
print(f'OK: shape={v.shape}')
"

# 验证 reranker
python -c "
from sentence_transformers import CrossEncoder
m = CrossEncoder('models/bge-reranker-v2-m3')
score = m.predict([('query', 'doc')])
print(f'OK: score={score[0]:.4f}')
"
```

## 在代码中使用本地模型

教程代码会自动检测本地模型 (如存在), 优先使用; 否则下载:

```python
from sentence_transformers import SentenceTransformer
import os

# 自动选本地或下载
local_path = "models/bge-small-zh-v1.5"
if os.path.isdir(local_path) and os.path.exists(f"{local_path}/config.json"):
    model = SentenceTransformer(local_path)
    print(f"[OK] using local model: {local_path}")
else:
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    print("[WARN] local not found, using remote (will download)")
```

## 磁盘空间

| 模型 | 大小 | 推荐场景 |
|------|------|---------|
| bge-small-zh-v1.5 | 100MB | 任意电脑 |
| bge-reranker-v2-m3 | 600MB | 任意电脑 |
| Qwen2.5-0.5B-Instruct | 1GB | 推荐 (4GB+ 内存) |
| Qwen2.5-7B-Instruct | 15GB | 进阶 (16GB+ 内存, GPU 推荐) |
| Qwen2.5-72B-Instruct | 140GB | 仅 GPU 服务器 |

## 国内源速度对比 (实测)

| 源 | bge-small (100MB) | Qwen2.5-0.5B (1GB) | 备注 |
|----|---|---|---|
| ModelScope | ~30s | ~5min | 国内 CDN, 稳定 |
| hf-mirror.com | ~1min | ~10min | HF 镜像, 偶尔抽风 |
| HuggingFace 直连 | ~5min+ | ~30min+ | 海外用户才能用, 国内经常 timeout |

**强烈推荐国内用户用 ModelScope**.
