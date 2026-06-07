# Docker 部署指南 (Wave 14-C)

> 5 分钟在 Docker 中跑通整个教程环境, 包含所有依赖 + 中间件 + 真实 LLM 调用.
> 镜像已配置国内源 (清华 pip / 阿里云 Docker Registry / HF 镜像), 国内访问速度 5-10 MB/s.

## 1. 快速开始 (3 步)

### Step 1: 准备 `.env`

```bash
# 在仓库根目录
cp .env.dockerexample .env
# 编辑 .env, 填入至少 1 个 API Key
# DEEPSEEK_API_KEY=sk-xxxxxxxxxx
```

### Step 2: 构建并启动

```bash
# 仅核心 (无中间件, ~3 min 构建)
make -C code docker-build
make -C code docker-up

# 包含 Redis (LangGraph checkpoint 用, ~3 min)
make -C code docker-llm

# GPU 模式 (需 NVIDIA GPU + nvidia-container-toolkit)
make -C code docker-gpu
```

### Step 3: 进入容器使用

```bash
# 临时进容器
make -C code docker-bash

# 容器内:
cd /app/code
make ci-quick           # 验证 5 项检查
make llm-doctor         # 诊断 API Key
python ch12_*/core/01_*.py    # 跑例子
```

## 2. 三种 profile 详解

### `core` (默认, 推荐先试)

```bash
docker compose --profile core up -d
```

- 仅 1 个容器: `llm-kb-app`
- 包含: Python 3.12 + core tier + llm tier 依赖
- 不含: Redis / Postgres (无状态服务)
- 适用: 个人学习, 跑通教程例子

### `llm` (进阶)

```bash
docker compose --profile llm up -d
```

- 2 个容器: `llm-kb-app` + `redis`
- 额外: Redis 7 (用于 LangGraph checkpoint / Langfuse cache)
- 适用: 想用持久化 agent state, 或跑 Ch20 LLMOps 例子

### `gpu` (生产)

```bash
docker compose --profile gpu up -d
```

- 3 个容器: `app` + `redis` + `postgres (pgvector)`
- 额外: pgvector 扩展 (RAG 向量存储)
- 前提: 主机有 NVIDIA GPU + 安装 `nvidia-container-toolkit`
- 适用: 跑 Ch16/19/25/26 的 GPU 例子

## 3. 国内源加速 (推荐国内用户)

Docker 默认配置已优化:

| 资源 | 国内源 |
|------|--------|
| Docker base image | `docker.io/library/python:3.12-slim` (可换阿里云) |
| pip | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| HuggingFace | `https://hf-mirror.com` |
| ModelScope | 内置 (无需镜像) |

如需切换阿里云 Docker Registry (更快):

```bash
# .env
DOCKER_REGISTRY=registry.cn-hangzhou.aliyuncs.com

# 重新构建
make -C code docker-build
```

## 4. 数据持久化

| 数据 | 路径 | 持久化方式 |
|------|------|----------|
| 用户下载的模型 | `code/models/` | host volume (./code/models) |
| HuggingFace 缓存 | `~/.cache/huggingface/` | Docker volume `app-cache` |
| Redis 数据 | `/data` (容器内) | Docker volume `redis-data` |
| Postgres 数据 | `/var/lib/postgresql/data` | Docker volume `pg-data` |

**注意**: 删除 volume 会清空数据. 备份:

```bash
docker run --rm -v llm-knowledge-base_pg-data:/from -v $(pwd):/to alpine tar czf /to/pg-backup.tar.gz /from
```

## 5. 常见操作

### 查看日志

```bash
docker compose --profile core logs -f app
docker compose --profile llm logs -f redis
```

### 重启某个服务

```bash
docker compose --profile llm restart app
```

### 跑一次性命令

```bash
# 在容器内跑 llm_doctor
docker compose --profile core run --rm app python scripts/llm_doctor.py

# 跑 make ci
docker compose --profile core run --rm app make -C code ci-quick

# 下模型
docker compose --profile core run --rm app python scripts/download_models.py --all
```

### 清理

```bash
# 停所有
make -C code docker-down

# 停 + 删容器 (保留 volume)
make -C code docker-down && docker compose rm

# 停 + 删所有 (清空数据)
make -C code docker-down && docker volume rm llm-knowledge-base_app-cache llm-knowledge-base_redis-data
```

## 6. 故障排查

### `docker build` 超时

国内网络拉 docker.io 慢. 解决方案:

```bash
# 方案 1: 切换阿里云
echo "DOCKER_REGISTRY=registry.cn-hangzhou.aliyuncs.com" >> .env
make -C code docker-build

# 方案 2: 使用 buildkit
DOCKER_BUILDKIT=1 make -C code docker-build
```

### `pip install` 在容器内失败

已在 Dockerfile 中设置清华源, 但如果你在 `make docker-build` 中看到 pypi 慢, 检查 build arg:

```bash
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
```

### `make ci` 在容器内 OOM

容器默认无内存限制. 如报 OOM:

```bash
# docker-compose.yml 中限制
services:
  app:
    deploy:
      resources:
        limits:
          memory: 4G
```

### 健康检查失败

```bash
docker inspect --format='{{.State.Health.Status}}' llm-kb-app
# 应输出: healthy

# 手动跑 healthcheck
docker exec llm-kb-app /usr/local/bin/healthcheck.sh
```

## 7. K8s 部署 (可选)

把 compose 转 K8s:

```bash
# 用 kompose (Docker 官方工具)
kompose convert -f docker-compose.yml

# 生成 deployment.yaml + service.yaml
kubectl apply -f .
```

需要的 K8s 资源:
- Deployment (app)
- Service (ClusterIP)
- ConfigMap (环境变量)
- Secret (API Keys)
- PVC (models 目录)

## 8. 镜像发布 (可选)

CI 自动 build + push 到 GitHub Container Registry:

```yaml
# .github/workflows/docker.yml (未来添加)
- name: Build and push
  uses: docker/build-push-action@v5
  with:
    push: true
    tags: ghcr.io/beefwrap4/llm-kb:latest
```

用户拉取:

```bash
docker pull ghcr.io/beefwrap4/llm-kb:latest
```

## 9. 资源占用

| Profile | 镜像 | 运行时内存 | 磁盘 |
|---------|------|----------|------|
| core | ~1.5GB | ~1GB | +700MB (模型) |
| llm | + Redis ~50MB | +50MB | +1GB (Redis) |
| gpu | + pgvector ~300MB | +200MB | +1GB (Postgres) |

总计: 任意 profile 都在 4GB 内存以内可跑.
