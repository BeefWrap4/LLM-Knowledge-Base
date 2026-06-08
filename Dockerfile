# ---
# Dockerfile (Wave 14-C) — Multi-stage build for LLM-Knowledge-Base
# Build:  docker build -t llm-kb:latest .
# Run:    docker run --rm -it -v ${PWD}/code:/app/code -v ${PWD}/.env:/app/.env:ro llm-kb:latest bash
# ---
# ════════════════════════════════════════════════════════════
# Stage 1: builder — 安装所有依赖 (cached)
# ════════════════════════════════════════════════════════════
ARG PYTHON_VERSION=3.12
ARG REGISTRY=docker.io
FROM ${REGISTRY}/library/python:${PYTHON_VERSION}-slim AS builder

# 国内 pip 镜像加速 (build 时可通过 --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple 覆盖)
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /build

# 系统依赖 (curl 用于 healthcheck, git 用于 ModelScope 备用下载)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements (利用 Docker layer cache, 依赖不变时不重新安装)
COPY code/requirements-core.txt code/requirements-llm.txt code/requirements-gpu.txt /build/

# 安装 core + llm (默认), GPU 单独 ARG
# pypi.org 作主, Tsinghua 作 fallback (Tsinghua 漏包如 google-generativeai, pip 单 index-url 不会自动回退)
RUN pip config set global.index-url https://pypi.org/simple \
    && pip config set global.extra-index-url ${PIP_INDEX_URL} \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /build/requirements-llm.txt

# 安装国内源 helper (ModelScope / huggingface_hub)
RUN pip install --no-cache-dir modelscope huggingface_hub

# ════════════════════════════════════════════════════════════
# Stage 2: runtime — 精简镜像
# ════════════════════════════════════════════════════════════
FROM ${REGISTRY}/library/python:${PYTHON_VERSION}-slim AS runtime

LABEL maintainer="BeefWrap4" \
      description="LLM Knowledge Base - 2026 LLM Interview Tutorial" \
      version="14.0"

# 运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

# 从 builder 复制 site-packages
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# 复制应用代码 (可被 -v 挂载覆盖)
COPY code/ /app/code/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/healthcheck.sh

# 默认环境变量 (用户可通过 -e 或 .env 覆盖)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLM_PROVIDER=deepseek \
    PATH="/app/code:/usr/local/bin:${PATH}"

# 创建空 models 目录 (用户可 -v 挂载本地下载的模型)
RUN mkdir -p /app/code/models && chown -R app:app /app

USER app

EXPOSE 8000 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
