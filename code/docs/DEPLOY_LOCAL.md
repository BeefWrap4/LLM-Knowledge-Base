# 本地环境部署完整指南 (Wave 26)

> 5 步把教程从"能跑"升级到"真用": 真实模型 + 真实中间件 + 真实 LLM, 0 打桩代码.
> 全部组件 ~3GB (含 1GB Qwen + 700MB embedding + 1GB Redis/pgvector), 适合笔记本.

## 1. 系统要求

| 资源 | 最低 | 推荐 |
|------|------|------|
| 磁盘 | 5 GB | 10 GB (含更多模型) |
| 内存 | 8 GB | 16 GB (含 7B 模型推理) |
| GPU | 可选 (CPU 可跑 0.5B) | NVIDIA 8GB+ (跑 7B) |
| 网络 | 国内源即可 (ModelScope / 清华) | — |

## 2. 5 步部署

### Step 1: 安装 Python 依赖

```bash
cd code/
make install-llm           # 30-60s, ~3GB
pip install llama-index-llms-openai-like   # 0.5s, LlamaIndex 0.14+ 需要
```

### Step 2: 配置 LLM API Key (至少 1 家)

```bash
cp .env.example .env
# 编辑 .env, 填入至少 1 个:
#   DEEPSEEK_API_KEY=sk-xxx   (推荐, ¥1/百万 token)
#   KIMI_API_KEY=sk-xxx
#   SILICONFLOW_API_KEY=sk-xxx  (Qwen 7B 免费)
#   MINIMAX_API_KEY=sk-cp-xxx  (Codin Plan)
```

### Step 3: 启动中间件 (Redis + pgvector)

```bash
# 避免与系统已有 Redis 冲突, 用 16379/15432 端口
docker run -d --name llm-kb-redis -p 16379:6379 --restart unless-stopped \
  redis:7-alpine redis-server --save 60 1 --appendonly yes

docker run -d --name llm-kb-postgres -p 15432:5432 \
  -e POSTGRES_USER=llmkb -e POSTGRES_PASSWORD=llmkb_test -e POSTGRES_DB=vectordb \
  pgvector/pgvector:pg16
```

(或用 `docker compose --profile llm up -d`, 已在 compose.yml 配置)

### Step 4: 下载真实模型 (国内源加速)

```bash
make download-models        # 默认: bge-small-zh + bge-reranker, ~700MB, 5 min
# 或
make download-models ARGS="--all"   # 含 Qwen 0.5B (~1.7GB)
```

下载目录: `code/models/`
- `bge-small-zh-v1.5/` (184MB) — RAG embedding
- `bge-reranker-v2-m3/` (2.2GB) — RAG 二次排序  
- `Qwen2.5-0.5B-Instruct/` (954MB) — 小型 LLM (本地推理)

### Step 5: 验证集成

```bash
# 4 项真实集成测试 (Embedding + Redis + pgvector + 4 厂商 LLM)
python scripts/test_integration.py

# 期望输出:
#   [✓] embedding  (bge-small-zh-v1.5)
#   [✓] redis      (localhost:16379)
#   [✓] pgvector   (localhost:15432)
#   [✓] llm        (4 厂商)
#   🎉 全部通过!
```

## 3. 跑真实 LLM 例子

```bash
# 13 个真实 LLM 例子一键跑 (~90s, 扣费 ~¥0.01)
bash scripts/run_real_demos.sh

# 切换厂商
bash scripts/run_real_demos.sh MiniMax
bash scripts/run_real_demos.sh deepseek
```

## 4. 跑 RAG 例子 (用本地 bge 模型)

```bash
cd code/

# 默认 mock 模式 (任何电脑可跑)
python ch14_rag/llm/01_rag_indexing_pipeline.py

# 真实模式 (需本地 bge 模型)
# RAG 例子会自动检测 models/bge-small-zh-v1.5, 优先本地加载
python ch14_rag/llm/01_rag_indexing_pipeline.py
```

## 5. 跑 Reranker 例子

```bash
python ch14_rag/llm/14_reranker_advanced_rag.py
# 输出: top-3 重排结果 (cosine 分数排序)
```

## 6. 跑 Qwen 0.5B 本地推理

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tok = AutoTokenizer.from_pretrained('code/models/Qwen2.5-0.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('code/models/Qwen2.5-0.5B-Instruct')
text = tok.apply_chat_template([{'role':'user','content':'你好'}], tokenize=False, add_generation_prompt=True)
out = m.generate(**tok(text, return_tensors='pt'), max_new_tokens=80, do_sample=False)
print(tok.decode(out[0][3:], skip_special_tokens=True))
"
# 输出: 你好, 我是 Qwen...
```

## 7. pgvector 用法示例 (RAG 向量库)

```python
import psycopg2
from sentence_transformers import SentenceTransformer

m = SentenceTransformer('code/models/bge-small-zh-v1.5')

c = psycopg2.connect(host='localhost', port=15432, user='llmkb',
                     password='llmkb_test', dbname='vectordb')
c.autocommit = True
cur = c.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
cur.execute('CREATE TABLE docs (id SERIAL PRIMARY KEY, content TEXT, embedding VECTOR(512))')

# 插入
emb = m.encode(['Python 教程'], normalize_embeddings=True)
vec_str = '[' + ','.join('%.7f' % x.item() for x in emb.flatten()) + ']'
cur.execute('INSERT INTO docs (content, embedding) VALUES (%s, %s)', ('Python 教程', vec_str))

# 查询
cur.execute("""
    SELECT content, 1 - (embedding <=> %s::vector) AS cosine
    FROM docs ORDER BY embedding <=> %s::vector LIMIT 1
""", (vec_str, vec_str))
print(cur.fetchone())  # ('Python 教程', ~1.0)
```

## 8. 故障排查

### Redis 连不上

```bash
# 检查容器运行状态
docker ps | grep llm-kb-redis
docker logs llm-kb-redis

# 重新启动
docker restart llm-kb-redis
```

### pgvector type "vector" does not exist

```bash
# 在数据库中启用扩展
docker exec -it llm-kb-postgres psql -U llmkb -d vectordb -c "CREATE EXTENSION vector;"
```

### 模型下载失败 (网络问题)

```bash
# 切换国内源
export HF_ENDPOINT=https://hf-mirror.com
python code/scripts/download_models.py --embedding
```

### LLM API 返回 401

```bash
# 检查 .env 文件名 (不是 .env.example) 和 Key 格式
cat code/.env
# 重新跑
python code/scripts/llm_doctor.py
```

## 9. 卸载 (清理)

```bash
# 停中间件
docker stop llm-kb-redis llm-kb-postgres
docker rm llm-kb-redis llm-kb-postgres

# 删模型 (~3GB 释放)
rm -rf code/models/bge-* code/models/Qwen*

# 删 .env (含 API Key)
rm code/.env
```

## 10. 完整组件清单

```
本地部署栈 (Wave 26):
├── Python 3.12 (miniconda py312)
├── 真实模型 (~3GB)
│   ├── bge-small-zh-v1.5 (184MB, embedding)
│   ├── bge-reranker-v2-m3 (2.2GB, reranker)
│   └── Qwen2.5-0.5B-Instruct (954MB, small LLM)
├── 中间件
│   ├── Redis 7 (localhost:16379, LangGraph checkpoint + cache)
│   └── pgvector 0.8.2 (localhost:15432, RAG 向量库)
├── LLM API (4 厂商任选)
│   ├── DeepSeek (deepseek-chat V3)
│   ├── Kimi (moonshot-v1-8k)
│   ├── SiliconFlow (Qwen 2.5 7B)
│   └── MiniMax (MiniMax-Text-01)
└── code/ 伴侣 (439 .py, 100% 通过)

一键验证: python code/scripts/test_integration.py
一键跑真实例子: bash code/scripts/run_real_demos.sh
```
