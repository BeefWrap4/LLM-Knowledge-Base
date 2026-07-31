#!/bin/bash
# ---
# code/scripts/setup_local.sh (Wave 27)
# 本地环境设置: 装依赖 + 启中间件 + 创建配置模板 + 可选下载模型
# Usage:  bash code/scripts/setup_local.sh
#         bash code/scripts/setup_local.sh --skip-models   # 不下模型
#         bash code/scripts/setup_local.sh --all-models    # 显式确认全部注册模型
#         bash code/scripts/setup_local.sh --skip-test      # 不跑离线验证
#         bash code/scripts/setup_local.sh --real-integration  # 显式运行计费集成测试
# ---
# 完整部署文档: code/docs/DEPLOY_LOCAL.md
# 耗时取决于网络、镜像缓存、模型和本机资源，不作固定承诺。

set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)

# 探测 make 路径 (Git Bash on Windows 不在 PATH)
if ! command -v make >/dev/null 2>&1; then
    if [ -x "/c/Program Files/Git/usr/bin/make.exe" ]; then
        export PATH="/c/Program Files/Git/usr/bin:$PATH"
    elif [ -x "/usr/bin/make.exe" ]; then
        export PATH="/usr/bin:$PATH"
    fi
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  LLM-Knowledge-Base 本地环境设置 (5 步)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────
# Step 1: 装 Python 依赖 (直接调 pip, 不依赖 make)
# ─────────────────────────────────────────────────────────
echo "[1/5] 装 Python 依赖（耗时取决于缓存与网络）..."
if command -v make >/dev/null 2>&1; then
    make install-llm
else
    echo "  (make 未装, 直接用 pip)"
    pip install -r requirements-core.txt 2>&1 | tail -1
    pip install -r requirements-llm.txt 2>&1 | tail -1
fi
# 额外: llama_index 0.14+ 需要 llama-index-llms-openai-like
pip install llama-index-llms-openai-like 2>&1 | tail -1
echo ""

# ─────────────────────────────────────────────────────────
# Step 2: 启中间件 (Redis + pgvector)
# ─────────────────────────────────────────────────────────
echo "[2/5] 启动中间件 (Redis + pgvector)..."

# 检查 Redis 是否已运行
if docker ps --format '{{.Names}}' | grep -q '^llm-kb-redis$'; then
    echo "  ✓ llm-kb-redis 已在运行"
else
    docker run -d --name llm-kb-redis -p 16379:6379 --restart unless-stopped \
        redis:7-alpine redis-server --save 60 1 --appendonly yes 2>&1 | tail -1
    echo "  ✓ llm-kb-redis 启动 (localhost:16379)"
fi

# 检查 pgvector 是否已运行
if docker ps --format '{{.Names}}' | grep -q '^llm-kb-postgres$'; then
    echo "  ✓ llm-kb-postgres 已在运行"
else
    docker run -d --name llm-kb-postgres -p 15432:5432 \
        -e POSTGRES_USER=llmkb -e POSTGRES_PASSWORD=llmkb_test -e POSTGRES_DB=vectordb \
        pgvector/pgvector:pg16 2>&1 | tail -1
    echo "  ✓ llm-kb-postgres 启动 (localhost:15432)"
fi

sleep 3
echo ""

# ─────────────────────────────────────────────────────────
# Step 3: 配置 .env (若不存在)
# ─────────────────────────────────────────────────────────
echo "[3/5] 配置 .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  ✓ .env.example → .env 复制完成"
    echo "  ⚠️  请编辑 .env 填入至少 1 个 API Key (推荐 DEEPSEEK_API_KEY)"
    echo "     填好后: vim .env  (或任何编辑器)"
else
    echo "  ✓ .env 已存在 (跳过复制)"
fi
echo ""

# ─────────────────────────────────────────────────────────
# Step 4: 下载模型 (可跳过)
# ─────────────────────────────────────────────────────────
SKIP_MODELS=0
ALL_MODELS=0
for arg in "$@"; do
    [ "$arg" = "--skip-models" ] && SKIP_MODELS=1
    [ "$arg" = "--all-models" ] && ALL_MODELS=1
done

if [ $SKIP_MODELS -eq 1 ]; then
    echo "[4/5] 跳过模型下载 (--skip-models)"
elif [ $ALL_MODELS -eq 1 ]; then
    echo "[4/5] 下载全部注册模型（已通过 --all-models 显式确认大型/受许可项）..."
    python scripts/download_models.py --all --confirm-large
else
    echo "[4/5] 仅下载默认 required 模型（体积与耗时以当前 revision/网络为准）..."
    python scripts/download_models.py --required-only
fi
echo ""

# ─────────────────────────────────────────────────────────
# Step 5: 集成测试 (可跳过)
# ─────────────────────────────────────────────────────────
SKIP_TEST=0
REAL_INTEGRATION=0
for arg in "$@"; do
    [ "$arg" = "--skip-test" ] && SKIP_TEST=1
    [ "$arg" = "--real-integration" ] && REAL_INTEGRATION=1
done

if [ $SKIP_TEST -eq 1 ]; then
    echo "[5/5] 跳过验证 (--skip-test)"
else
    if [ $REAL_INTEGRATION -eq 1 ]; then
        echo "[5/5] 显式运行真实集成测试（Redis + pgvector + 一个计费 LLM）..."
        TEST_COMMAND=(env RUN_REAL_INTEGRATION=1 LLM_MOCK=0 python scripts/test_integration.py)
    else
        echo "[5/5] 默认离线验证（不读 Key、不调用真实 API）..."
        TEST_COMMAND=(env LLM_MOCK=1 python scripts/verify_all.py)
    fi
    if "${TEST_COMMAND[@]}"; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  设置流程完成；上方结果只覆盖本次实际执行的检查。"
        echo ""
        echo "  下一步:"
        echo "    LLM_MOCK=1 python scripts/run_all_examples.py --tier llm"
        echo "    bash scripts/run_real_demos.sh --confirm-real quick deepseek"
        echo "    make ci"
        echo "═══════════════════════════════════════════════════════════════"
    else
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  ⚠️  验证失败, 排查后再跑."
        echo "  详细: code/docs/DEPLOY_LOCAL.md §9 故障排查"
        echo "═══════════════════════════════════════════════════════════════"
        exit 1
    fi
fi
