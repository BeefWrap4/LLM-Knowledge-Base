#!/usr/bin/env bash
# Explicit, billable real-API smoke runner.
#
# Examples:
#   bash scripts/run_real_demos.sh --confirm-real quick deepseek
#   bash scripts/run_real_demos.sh --confirm-real all MiniMax --parallel 2
#
# This script never falls back to mock. A zero exit code means every selected
# example either made a non-mock call successfully or was reported separately
# as [SKIP] because an optional framework dependency was unavailable.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

PARALLEL=1
PROVIDER="${LLM_PROVIDER:-}"
MODE="all"
CONFIRM_REAL=0
NEXT_IS_PARALLEL=0

for arg in "$@"; do
    if [ "$NEXT_IS_PARALLEL" = "1" ]; then
        PARALLEL="$arg"
        NEXT_IS_PARALLEL=0
        continue
    fi
    case "$arg" in
        --confirm-real)
            CONFIRM_REAL=1
            ;;
        --parallel)
            NEXT_IS_PARALLEL=1
            ;;
        --parallel=*)
            PARALLEL="${arg#--parallel=}"
            ;;
        quick|all)
            MODE="$arg"
            ;;
        deepseek|kimi|siliconflow|minimax|MiniMax|openai|anthropic)
            PROVIDER="$arg"
            ;;
        *)
            echo "未知参数: $arg"
            echo "用法: $0 --confirm-real [quick|all] [provider] [--parallel N]"
            exit 2
            ;;
    esac
done

if [ "$NEXT_IS_PARALLEL" = "1" ]; then
    echo "--parallel 缺少正整数参数"
    exit 2
fi
case "$PARALLEL" in
    ''|*[!0-9]*|0)
        echo "--parallel 必须是正整数"
        exit 2
        ;;
esac

if [ "$CONFIRM_REAL" != "1" ]; then
    echo "拒绝运行：真实 API 会发送数据并产生费用。"
    echo "确认后添加 --confirm-real；默认离线验收请使用："
    echo "  python scripts/run_all_examples.py --tier llm"
    exit 2
fi

if [ -z "$PROVIDER" ]; then
    if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
        PROVIDER="deepseek"
    elif [ -n "${KIMI_API_KEY:-}" ]; then
        PROVIDER="kimi"
    elif [ -n "${SILICONFLOW_API_KEY:-}" ]; then
        PROVIDER="siliconflow"
    elif [ -n "${MINIMAX_API_KEY:-}" ]; then
        PROVIDER="MiniMax"
    elif [ -n "${OPENAI_API_KEY:-}" ]; then
        PROVIDER="openai"
    elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        PROVIDER="anthropic"
    else
        echo "未检测到任何支持厂商的 API Key；不会降级到 mock。"
        exit 2
    fi
fi

case "${PROVIDER,,}" in
    deepseek)
        REQUIRED_KEY="DEEPSEEK_API_KEY"
        PROVIDER="deepseek"
        ;;
    kimi)
        REQUIRED_KEY="KIMI_API_KEY"
        PROVIDER="kimi"
        ;;
    siliconflow)
        REQUIRED_KEY="SILICONFLOW_API_KEY"
        PROVIDER="siliconflow"
        ;;
    minimax)
        REQUIRED_KEY="MINIMAX_API_KEY"
        PROVIDER="MiniMax"
        ;;
    openai)
        REQUIRED_KEY="OPENAI_API_KEY"
        PROVIDER="openai"
        ;;
    anthropic)
        REQUIRED_KEY="ANTHROPIC_API_KEY"
        PROVIDER="anthropic"
        ;;
    *)
        echo "不支持的 provider: $PROVIDER"
        exit 2
        ;;
esac

if [ -z "${!REQUIRED_KEY:-}" ]; then
    echo "LLM_PROVIDER=$PROVIDER 需要环境变量 $REQUIRED_KEY；不会改用其他 Key 或 mock。"
    exit 2
fi

export LLM_PROVIDER="$PROVIDER"
export LLM_MOCK=0
unset USE_REAL_API || true

QUICK_EXAMPLES=(
    "ch13_prompt_engineering/llm/06_self_consistency_cot.py"
    "ch13_prompt_engineering/llm/09_compare_temperatures.py"
    "ch15_agent/llm/02_react_agent_from_scratch.py"
)

ALL_EXAMPLES=(
    "${QUICK_EXAMPLES[@]}"
    "ch17_evaluation/llm/05_llm_as_judge.py"
    "ch18_llm_frameworks/llm/02_llmchain_basic.py"
    "ch18_llm_frameworks/llm/03_sequential_chain.py"
    "ch18_llm_frameworks/llm/05_conversation_buffer_memory.py"
    "ch18_llm_frameworks/llm/09_chatbot_with_memory.py"
    "ch18_llm_frameworks/llm/13_llamaindex_vectorstore_index.py"
    "ch18_llm_frameworks/llm/14_llamaindex_summary_index.py"
)

if [ "$MODE" = "quick" ]; then
    EXAMPLES=("${QUICK_EXAMPLES[@]}")
else
    EXAMPLES=("${ALL_EXAMPLES[@]}")
fi

# These examples exercise OpenAI-specific APIs and are not provider-neutral.
if [ "$MODE" = "all" ] && [ "$PROVIDER" = "openai" ]; then
    EXAMPLES+=(
        "ch13_prompt_engineering/llm/14_openai_auto_caching.py"
        "ch13_prompt_engineering/llm/20_openai_json_schema_strict.py"
    )
fi

TOTAL=${#EXAMPLES[@]}
TMPDIR_RUN=$(mktemp -d)
cleanup() {
    if [ -n "${TMPDIR_RUN:-}" ] && [ -d "$TMPDIR_RUN" ]; then
        rm -rf -- "$TMPDIR_RUN"
    fi
}
trap cleanup EXIT

run_one() {
    local rel="$1"
    local idx="$2"
    local script="$ROOT/$rel"
    local outfile="$TMPDIR_RUN/$idx.out"
    local statusfile="$TMPDIR_RUN/$idx.status"
    local started
    local finished
    local elapsed
    started=$(date +%s)

    if [ ! -f "$script" ]; then
        printf 'FAIL|0|%s|文件不存在\n' "$rel" > "$statusfile"
        return
    fi

    if python "$script" >"$outfile" 2>&1; then
        finished=$(date +%s)
        elapsed=$((finished - started))
        if grep -Eq '\[SKIP\]' "$outfile"; then
            printf 'SKIP|%s|%s|可选条件未满足\n' "$elapsed" "$rel" > "$statusfile"
        elif grep -Eiq '\[mock\]|mock[=:][[:space:]]*true|离线(演示|模式)' "$outfile"; then
            printf 'FAIL|%s|%s|检测到 mock/离线输出，不能计为真实验收\n' \
                "$elapsed" "$rel" > "$statusfile"
        else
            printf 'PASS|%s|%s|non-mock process exit 0\n' "$elapsed" "$rel" > "$statusfile"
        fi
    else
        finished=$(date +%s)
        elapsed=$((finished - started))
        local err
        err=$(grep -E 'Error|ERROR|Traceback|Exception' "$outfile" 2>/dev/null | head -1 | head -c 120 || true)
        printf 'FAIL|%s|%s|%s\n' "$elapsed" "$rel" "${err:-process exit non-zero}" > "$statusfile"
    fi
}

echo "真实 API 条件性验收：provider=$PROVIDER, mode=$MODE, parallel=$PARALLEL"
echo "选中 $TOTAL 个示例；实际费用与请求数据以厂商控制台为准。"

active=0
idx=0
for rel in "${EXAMPLES[@]}"; do
    idx=$((idx + 1))
    run_one "$rel" "$idx" &
    active=$((active + 1))
    if [ "$active" -ge "$PARALLEL" ]; then
        wait
        active=0
    fi
done
wait

PASS=0
FAIL=0
SKIP=0
for idx in $(seq 1 "$TOTAL"); do
    statusfile="$TMPDIR_RUN/$idx.status"
    if [ ! -f "$statusfile" ]; then
        echo "FAIL|0|unknown|缺少状态文件" > "$statusfile"
    fi
    IFS='|' read -r status elapsed rel detail < "$statusfile"
    printf '[%2d/%d] %-4s %-58s %4ss  %s\n' \
        "$idx" "$TOTAL" "$status" "$(basename "$rel")" "$elapsed" "$detail"
    case "$status" in
        PASS) PASS=$((PASS + 1)) ;;
        SKIP) SKIP=$((SKIP + 1)) ;;
        *) FAIL=$((FAIL + 1)) ;;
    esac
done

echo "完成：$PASS passed / $SKIP skipped / $FAIL failed / $TOTAL total"
echo "PASS 仅表示该进程未检测到 mock 且正常退出；仍需核对 provider usage、响应模型和账单。"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
