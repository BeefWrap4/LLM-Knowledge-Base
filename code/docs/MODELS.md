# 模型下载与使用指南 (Wave 14-B)

> 教程中部分例子需要真实的模型权重 (Ch14 RAG embedding, Ch16 推理 mock, Ch22 语义去重).
> 本指南说明: 选哪个模型 / 在哪下载 / 国内源加速 / 代码中如何引用.

## 1. 快速选型

| 章节 | 模型 | 大小 | 必需? | 用途 |
|------|------|------|------|------|
| Ch14 RAG 基础 | `BAAI/bge-small-zh-v1.5` | 100MB | **必需** | 句向量, 召回 |
| Ch14 RAG 高级 | `BAAI/bge-reranker-v2-m3` | 600MB | 推荐 | 二次排序 |
| Ch16 微调 | `Qwen/Qwen2.5-0.5B-Instruct` | 1GB | 可选 | GRPO/ORPO mock 训练 |
| Ch22 数据去重 | `BAAI/bge-small-zh-v1.5` | 100MB | 复用 | 语义指纹 |
| Ch26 世界模型 | 任何视频生成 | - | 不需要 | 调用云 API (无本地) |
| Ch27 推理 | 任何 LLM | - | 不需要 | 调用云 API (DeepSeek R1) |
| Ch28 端侧 | Qwen2.5 GGUF | 4GB+ | 笔记本需要 | 4-bit 量化推理 |

**最小集 (~700MB)**: bge-small-zh-v1.5 + bge-reranker-v2-m3, 跑通 80% 例子.

## 2. 国内源下载

### 方案 A: ModelScope (推荐, 5-10 MB/s)

```bash
pip install modelscope
python code/scripts/download_models.py --all
# 或单个
python code/scripts/download_models.py --embedding
```

### 方案 B: HuggingFace 镜像 (1-3 MB/s)

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
python code/scripts/download_models.py --all
```

### 方案 C: ModelScope CLI 直接用

```bash
modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir code/models/bge-small-zh-v1.5
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir code/models/bge-reranker-v2-m3
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir code/models/Qwen2.5-0.5B-Instruct
```

## 3. 在代码中引用

### 自动检测本地 (推荐)

```python
import os
from sentence_transformers import SentenceTransformer

LOCAL_PATH = "models/bge-small-zh-v1.5"
if os.path.exists(f"{LOCAL_PATH}/config.json"):
    model = SentenceTransformer(LOCAL_PATH)        # 本地
else:
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")  # 远端 fallback
```

### 显式本地

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("code/models/bge-small-zh-v1.5")
```

### 教程代码已适配

Ch14/Ch22 的例子在加载时会自动:
1. 检查 `code/models/{model_name}/config.json` 是否存在
2. 存在 → 加载本地
3. 不存在 → 远端下载 (或 SKIP + 提示用户)

## 4. 验证下载

```bash
cd code/

# Embedding 测试
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('models/bge-small-zh-v1.5')
v = m.encode(['测试中文向量'])
print(f'OK: shape={v.shape}, dim={v.shape[-1]}')
"

# Reranker 测试
python -c "
from sentence_transformers import CrossEncoder
m = CrossEncoder('models/bge-reranker-v2-m3')
score = m.predict([('什么是 RAG', 'RAG 是检索增强生成')])
print(f'OK: relevance={score[0]:.4f}')
"
```

## 5. 速度对比 (实测 2026-06)

| 源 | bge-small (100MB) | bge-reranker (600MB) | Qwen2.5-0.5B (1GB) |
|----|---|---|---|
| ModelScope (CN) | **30s** | **3min** | **5min** |
| hf-mirror.com (CN) | 1min | 8min | 15min |
| HF 直连 (海外) | 2min | 12min | 20min |
| HF 直连 (CN, 需代理) | 5min+ | 30min+ | timeout |

## 6. 教程选择这些模型的理由

### 为什么是 `bge-small-zh-v1.5` (而不是 `bge-large` 或 OpenAI text-embedding-3)?

| 维度 | bge-small-zh | bge-large-zh | text-embedding-3-small |
|------|---|---|---|
| 大小 | 100MB | 1.3GB | API (需钱) |
| 维度 | 512 | 1024 | 1536 |
| 中文效果 | 优秀 | 略好 | 优秀 |
| **本教程选** | ✓ | (可选) | (需 API) |

- 100MB 让任意电脑 30 秒下载
- 512 维对教程例子足够
- 完全本地, 无需 API Key

### 为什么是 `Qwen2.5-0.5B` (而不是 7B / 72B)?

| 维度 | 0.5B | 1.5B | 7B | 72B |
|------|------|------|-----|-----|
| 大小 | 1GB | 3GB | 15GB | 140GB |
| 内存 | 2GB | 4GB | 16GB | 160GB+ GPU |
| 推理速度 (CPU) | 5 tok/s | 2 tok/s | <1 | 跑不动 |
| **本教程选** | ✓ | (备选) | (需 GPU) | (服务器) |

- 1GB 让 4GB 内存的笔记本也能跑
- 速度 5 tok/s, 教学演示足够
- 真实推理用云 API (DeepSeek R1) 即可

## 7. 故障排查

### `OSError: We couldn't connect to huggingface.co`

国内网络问题. 改用国内源:

```bash
# 方式 1: 用我们的脚本
python code/scripts/download_models.py --all

# 方式 2: 设置镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download ...
```

### `SentenceTransformer 加载本地失败: 不是有效模型`

目录结构不完整. 期望:
```
code/models/bge-small-zh-v1.5/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
├── pytorch_model.bin  (或 model.safetensors)
└── modules.json       (sentence-transformers 特有)
```

如果用 git LFS 部分下载, 可能缺文件. 重新下载:

```bash
rm -rf code/models/bge-small-zh-v1.5
python code/scripts/download_models.py --embedding
```

### 磁盘空间不足

`df -h` 检查. 至少预留 2GB.

## 8. 进阶: 微调自己的模型 (可选, 不在教程范围)

```bash
# 需要 16GB+ GPU
python ch16_finetuning/gpu/01_lora_finetuning.py --mock    # mock 演示
python ch16_finetuning/gpu/02_qlora_config.py --mock
```

实操需要 7B+ 模型 (本地 `code/models/Qwen2.5-7B-Instruct/` ~15GB) + CUDA GPU.
