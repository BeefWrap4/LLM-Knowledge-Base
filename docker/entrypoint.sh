#!/bin/sh
# ---
# docker/entrypoint.sh
# 容器启动入口: 等待依赖 + 打印欢迎信息 + 执行 CMD
# ---
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  LLM Knowledge Base - Docker Container"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 1. 加载 .env (如果存在)
if [ -f /app/.env ]; then
    echo "[entrypoint] 加载 /app/.env"
    set -a
    . /app/.env
    set +a
fi

# 2. 等待 Redis (如果 REDIS_URL 设置)
if [ -n "$REDIS_URL" ]; then
    echo "[entrypoint] 等待 Redis: $REDIS_URL"
    # 简化: 假设 docker-compose 内服务名 'redis' 端口 6379
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if nc -z redis 6379 2>/dev/null || nc -z localhost 6379 2>/dev/null; then
            echo "  ✓ Redis 已就绪"
            break
        fi
        sleep 1
    done
fi

# 3. 打印诊断
echo ""
echo "[diagnostic] Python: $(python --version)"
echo "[diagnostic] 工作目录: $(pwd)"
echo "[diagnostic] LLM_PROVIDER: ${LLM_PROVIDER:-mock}"
echo "[diagnostic] DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:+***set***}${DEEPSEEK_API_KEY:-<unset>}"
echo "[diagnostic] KIMI_API_KEY: ${KIMI_API_KEY:+***set***}${KIMI_API_KEY:-<unset>}"
echo "[diagnostic] SILICONFLOW_API_KEY: ${SILICONFLOW_API_KEY:+***set***}${SILICONFLOW_API_KEY:-<unset>}"
echo ""

# 4. 跑一次 LLM doctor (如果 LLM_PROVIDER 已配置)
if [ -n "$DEEPSEEK_API_KEY" ] || [ -n "$KIMI_API_KEY" ] || [ -n "$SILICONFLOW_API_KEY" ]; then
    echo "[entrypoint] 跑 LLM doctor..."
    cd /app/code && python scripts/llm_doctor.py 2>&1 | head -10 || echo "  (doctor 失败, 但容器继续)"
    echo ""
fi

# 5. 如果是 default cmd, 给个 banner; 否则执行 CMD
if [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  容器已就绪. 常用命令:"
    echo "    cd /app/code"
    echo "    make ci-quick         # 10 项离线门禁 (~30s)"
    echo "    make llm-doctor       # 诊断 API Key"
    echo "    python ch15_transformer/core/01_scaled_dot_product_attention.py"
    echo "    python scripts/download_models.py --all   # 下载模型"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    exec "$@"
else
    exec "$@"
fi
