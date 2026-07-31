# Docker 本地环境指南

> Docker profile 只负责声明本地容器与依赖边界，不保证固定构建时间、网络速度、GPU 兼容或
> 真实 LLM 可用。默认先运行离线验收；真实 API、Redis、pgvector 与 GPU 路径分别验证。
> 使用第三方镜像/包/模型镜像前应核对来源、完整性与组织安全策略。

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
# 仅核心（无中间件；构建时间取决于缓存与网络）
make -C code docker-build
make -C code docker-up

# 包含 Redis（LangGraph checkpoint 等条件性集成）
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
make llm-doctor         # 只读查看 provider 配置，不联网
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

### `gpu`（本地条件性 GPU 环境）

```bash
docker compose --profile gpu up -d
```

- 3 个容器: `app` + `redis` + `postgres (pgvector)`
- 额外: pgvector 扩展 (RAG 向量存储)
- 本机测试端口: Redis `16379`、pgvector `15432`；可用 `REDIS_PORT` / `PG_PORT` 覆盖
- 前提: 主机有 NVIDIA GPU + 安装 `nvidia-container-toolkit`
- 适用: 在兼容的本地 NVIDIA 容器栈中逐项验证 Ch16/19/25/26 的条件路径
- 不代表生产安全、容量、持久化、可观测性或故障恢复已经验收

## 3. 软件源与镜像（按环境显式选择）

发布构建默认使用 PyPI 官方索引，避免第三方镜像的同步延迟或临时访问限制影响 CI。
如组织允许使用镜像，应在当前网络中先验证，再通过构建参数显式覆盖：

| 资源 | 默认值 / 可选方案 |
|------|-------------------|
| Docker base image | `docker.io/library/python:3.12-slim` (可换阿里云) |
| pip | 默认 `https://pypi.org/simple`；可用 `PIP_INDEX_URL` 覆盖 |
| HuggingFace | `https://hf-mirror.com` |
| ModelScope | 内置 (无需镜像) |

如组织允许使用阿里云 Docker Registry，可显式切换；速度不作保证：

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
docker compose --profile core run --rm app python scripts/download_models.py --required-only
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

先用默认官方索引重试并核对代理、DNS 与证书。如果所在组织提供经过验证的镜像，
再显式设置 build arg；镜像地址仅作示例，不保证当前可用：

```bash
docker build --build-arg PIP_INDEX_URL=https://your-approved-mirror.example/simple .
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
| core | 取决于基础镜像与依赖 revision | 取决于运行时 | 模型按当前清单另计 |
| llm | + Redis ~50MB | +50MB | +1GB (Redis) |
| gpu | + pgvector ~300MB | +200MB | +1GB (Postgres) |

总计: 任意 profile 都在 4GB 内存以内可跑.
