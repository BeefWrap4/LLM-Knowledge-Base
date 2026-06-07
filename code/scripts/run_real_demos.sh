#!/bin/bash
# ---
# code/scripts/run_real_demos.sh (Wave 22 + Wave 28)
# 一键跑 13 个真实 LLM 调用例子 (支持并行)
# Usage:
#   bash scripts/run_real_demos.sh                # 串行 (~90s, 默认)
#   bash scripts/run_real_demos.sh --parallel 4    # 并行 4 任务 (~45s)
#   bash scripts/run_real_demos.sh quick           # 仅 3 个核心 (~30s)
#   bash scripts/run_real_demos.sh MiniMax         # 指定厂商
#   bash scripts/run_real_demos.sh --parallel 4 MiniMax  # 组合
# ---
# 自动检查 API Key, 缺则降级 mock (但会提示)

set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)

# 探测 make 路径 (Git Bash on Windows 不在 PATH)
if ! command -v make >/dev/null 2>&1; then
    if [ -x "/c/Program Files/Git/usr/bin/make.exe" ]; then
        export PATH="/c/Program Files/Git/usr/bin:$PATH"
    fi
fi

# ─────────────────────────────────────────────────────────
# 解析参数
# ─────────────────────────────────────────────────────────
PARALLEL=1
PROVIDER=""
MODE=""
NEXT_IS_PARALLEL=0

for arg in "$@"; do
    if [ "$NEXT_IS_PARALLEL" = "1" ]; then
        PARALLEL="$arg"
        NEXT_IS_PARALLEL=0
        continue
    fi
    case "$arg" in
        --parallel)
            NEXT_IS_PARALLEL=1
            ;;
        --parallel=*)
            PARALLEL="${arg#--parallel=}"
            ;;
        --parallel[0-9]*)
            PARALLEL="${arg#--parallel}"
            ;;
        quick|all)
            MODE="$arg"
            ;;
        deepseek|kimi|siliconflow|MiniMax|openai|anthropic|mock)
            PROVIDER="$arg"
            ;;
        *)
            echo "未知参数: $arg"
            echo "用法: $0 [all|quick|<vendor>] [--parallel N]"
            exit 1
            ;;
    esac
done

MODE="${MODE:-all}"

# ─────────────────────────────────────────────────────────
# API Key 检查
# ─────────────────────────────────────────────────────────
HAS_KEY=0
[ -n "$DEEPSEEK_API_KEY" ] && HAS_KEY=1
[ -n "$KIMI_API_KEY" ] && HAS_KEY=1
[ -n "$SILICONFLOW_API_KEY" ] && HAS_KEY=1
[ -n "$MINIMAX_API_KEY" ] && HAS_KEY=1
[ -n "$OPENAI_API_KEY" ] && HAS_KEY=1

if [ $HAS_KEY -eq 0 ]; then
    echo "⚠️  未检测到任何 LLM API Key"
    echo "   设置至少 1 个后重试, e.g.:"
    echo "   export DEEPSEEK_API_KEY=sk-xxx"
    echo ""
    echo "   将以 mock 模式跑 (所有例子都 OK, 但内容是确定性响应)"
fi

# 默认厂商
if [ -z "$PROVIDER" ]; then
    PROVIDER="${LLM_PROVIDER:-deepseek}"
fi
export LLM_PROVIDER=$PROVIDER
export USE_REAL_API=1

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Real Demos — 13 个真实 LLM 调用例子"
echo "  provider=$PROVIDER, mode=$MODE, parallel=$PARALLEL"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────
# 选择例子集
# ─────────────────────────────────────────────────────────
case "$MODE" in
    quick)
        EXAMPLES=(
            "ch13_prompt_engineering/llm/06_self_consistency_cot.py"
            "ch13_prompt_engineering/llm/09_compare_temperatures.py"
            "ch17_evaluation/llm/05_llm_as_judge.py"
        )
        echo "模式: quick (3 个核心例子)"
        ;;
    all|"")
        EXAMPLES=(
            "ch13_prompt_engineering/llm/06_self_consistency_cot.py"
            "ch13_prompt_engineering/llm/09_compare_temperatures.py"
            "ch13_prompt_engineering/llm/14_openai_auto_caching.py"
            "ch13_prompt_engineering/llm/20_openai_json_schema_strict.py"
            "ch15_agent/llm/02_react_agent_from_scratch.py"
            "ch17_evaluation/llm/05_llm_as_judge.py"
            "ch17_evaluation/llm/12_langfuse_v3.py"
            "ch18_llm_frameworks/llm/02_llmchain_basic.py"
            "ch18_llm_frameworks/llm/05_conversation_buffer_memory.py"
            "ch18_llm_frameworks/llm/13_llamaindex_vectorstore_index.py"
            "ch18_llm_frameworks/llm/03_sequential_chain.py"
            "ch18_llm_frameworks/llm/09_chatbot_with_memory.py"
            "ch18_llm_frameworks/llm/14_llamaindex_summary_index.py"
        )
        echo "模式: all (全部 13 个)"
        ;;
    *)
        echo "未知 mode: $MODE"
        exit 1
        ;;
esac
echo ""

# ─────────────────────────────────────────────────────────
# 跑例子 (串行 or 并行)
# ─────────────────────────────────────────────────────────
TOTAL=${#EXAMPLES[@]}
START_TIME=$(date +%s)
TMPDIR_RUN=$(mktemp -d)
trap "rm -rf $TMPDIR_RUN" EXIT

run_one() {
    local rel="$1"
    local idx="$2"
    local script="$ROOT/$rel"
    local logfile="$TMPDIR_RUN/${idx}.log"
    local t0=$(date +%s)
    if [ ! -f "$script" ]; then
        echo "SKIP|$rel" > "$logfile"
        return
    fi
    if python "$script" > "$logfile" 2>&1; then
        local t1=$(date +%s)
        local elapsed=$((t1 - t0))
        local summary=$(grep -E "多数投票|答案|content|response|OK \(" "$logfile" 2>/dev/null | head -1 | head -c 60)
        echo "OK|$elapsed|$rel|$summary" > "$logfile"
    else
        local t1=$(date +%s)
        local elapsed=$((t1 - t0))
        local err=$(grep -E "Error|ERROR|Traceback" "$logfile" 2>/dev/null | head -1 | head -c 60)
        echo "FAIL|$elapsed|$rel|$err" > "$logfile"
    fi
}

if [ "$PARALLEL" -le 1 ]; then
    # 串行
    i=0
    for rel in "${EXAMPLES[@]}"; do
        i=$((i+1))
        printf "[%d/%d] %-65s ... " "$i" "$TOTAL" "$(basename "$rel")"
        logfile="$TMPDIR_RUN/${i}.log"
        t0=$(date +%s)
        if [ ! -f "$ROOT/$rel" ]; then
            echo "SKIP (文件不存在)"
            continue
        fi
        if python "$ROOT/$rel" > "$logfile" 2>&1; then
            t1=$(date +%s); elapsed=$((t1 - t0))
            summary=$(grep -E "多数投票|答案|content|response|OK \(" "$logfile" 2>/dev/null | head -1 | head -c 60)
            echo "OK (${elapsed}s) ${summary}"
            PASS=$((PASS+1))
        else
            t1=$(date +%s); elapsed=$((t1 - t0))
            echo "FAIL (${elapsed}s)"
            grep -E "Error|ERROR|Traceback" "$logfile" | head -2
            FAIL=$((FAIL+1))
        fi
    done
    END_TIME=$(date +%s)
    TOTAL_ELAPSED=$((END_TIME - START_TIME))
    PASS=$(grep -l "^OK|" $TMPDIR_RUN/*.log 2>/dev/null | wc -l)
    FAIL=$(grep -l "^FAIL|" $TMPDIR_RUN/*.log 2>/dev/null | wc -l)
    SKIP=$(grep -l "^SKIP|" $TMPDIR_RUN/*.log 2>/dev/null | wc -l)
else
    # 并行 (用 xargs -P 或 background jobs)
    echo "并行度: $PARALLEL"
    i=0
    pids=()
    for rel in "${EXAMPLES[@]}"; do
        i=$((i+1))
        # 后台启动
        ( run_one "$rel" "$i" ) &
        pids+=($!)
        # 控制并发度
        if [ ${#pids[@]} -ge $PARALLEL ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done
    # 等待所有后台任务
    wait
    END_TIME=$(date +%s)
    TOTAL_ELAPSED=$((END_TIME - START_TIME))
    # 收集结果
    PASS=0; FAIL=0; SKIP=0
    for logfile in $TMPDIR_RUN/*.log; do
        result=$(cat "$logfile" 2>/dev/null | head -1)
        idx=$(basename "$logfile" .log)
        rel=$(echo "$result" | cut -d'|' -f3)
        elapsed=$(echo "$result" | cut -d'|' -f2)
        summary=$(echo "$result" | cut -d'|' -f4-)
        if [[ "$result" == OK\|* ]]; then
            printf "  [%s/%d] OK   %-55s %3ds %s\n" "$idx" "$TOTAL" "$(basename "$rel")" "$elapsed" "${summary:0:40}"
            PASS=$((PASS+1))
        elif [[ "$result" == FAIL\|* ]]; then
            printf "  [%s/%d] FAIL %-55s %3ds %s\n" "$idx" "$TOTAL" "$(basename "$rel")" "$elapsed" "${summary:0:40}"
            FAIL=$((FAIL+1))
        else
            printf "  [%s/%d] SKIP %s\n" "$idx" "$TOTAL" "$rel"
            SKIP=$((SKIP+1))
        fi
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  完成: $PASS passed, $FAIL failed${SKIP:+ ($SKIP skipped)} (总计 $TOTAL, 耗时 ${TOTAL_ELAPSED}s)"
if [ "$PARALLEL" -gt 1 ]; then
    echo "  并行度: $PARALLEL (vs 串行 ~88s)"
fi
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  估算成本 (按 DeepSeek 价格):"
echo "    DeepSeek:    ¥1/百万 input tokens, ¥2/百万 output"
echo "    Kimi:        ¥12/百万 tokens"
echo "    SiliconFlow: 部分模型免费 (Qwen 7B), 其他 ¥1-4/百万"
echo "    MiniMax:  按订阅计划"
echo ""
echo "  本次运行约 10-50 个 chat 调用, 估计 ¥0.01 - ¥0.10"

exit $FAIL
