#!/bin/sh
# ---
# docker/healthcheck.sh
# 容器健康检查 (Docker HEALTHCHECK)
# ---
set -e

# 1. Python 解释器可用
python -c "import sys; print(sys.version)" >/dev/null

# 2. 关键包可导入 (langchain + sentence_transformers)
python -c "import langchain_core, langchain_openai" 2>/dev/null || \
    python -c "import langchain" 2>/dev/null || \
    echo "  [WARN] langchain 未安装"

# 3. code 目录可访问
test -d /app/code || exit 1

# 4. 至少 1 个 .py 例子可读
test -f /app/code/ch15_transformer/core/01_scaled_dot_product_attention.py || exit 1

# 5. (可选) Redis 健康
if [ -n "$REDIS_URL" ]; then
    nc -z redis 6379 2>/dev/null || true  # 不强制失败
fi

echo "OK"
exit 0
