#!/bin/bash
# ---
# code/scripts/run_real_demos.sh (Wave 22)
# 一键跑 13 个真实 LLM 调用例子
# Usage: bash code/scripts/run_real_demos.sh
#        bash code/scripts/run_real_demos.sh quick    # 只跑 3 个有代表性的
#        bash code/scripts/run_real_demos.sh deepseek  # 指定厂商
#        bash code/scripts/run_real_demos.sh MiniMax
# ---
# 自动检查 API Key, 缺则降级 mock (但会提示)

set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)

# 检测 API Key
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
PROVIDER=${LLM_PROVIDER:-deepseek}
export LLM_PROVIDER=$PROVIDER
export USE_REAL_API=1

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Real Demos — 13 个真实 LLM 调用例子 (provider=$PROVIDER)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 选择例子集
case "${1:-all}" in
    quick)
        EXAMPLES=(
            "ch13_prompt_engineering/llm/06_self_consistency_cot.py"
            "ch13_prompt_engineering/llm/09_compare_temperatures.py"
            "ch17_evaluation/llm/05_llm_as_judge.py"
        )
        echo "模式: quick (3 个核心例子)"
        ;;
    deepseek|kimi|siliconflow|MiniMax|openai)
        PROVIDER=$1
        export LLM_PROVIDER=$1
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
        echo "模式: provider=$PROVIDER (全部 13 个)"
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
        echo "未知参数: $1"
        echo "用法: $0 [all|quick|deepseek|kimi|siliconflow|MiniMax|openai]"
        exit 1
        ;;
esac
echo ""

# 跑每个例子
PASS=0
FAIL=0
TOTAL=${#EXAMPLES[@]}
START_TIME=$(date +%s)
i=0
for rel in "${EXAMPLES[@]}"; do
    i=$((i+1))
    script="$ROOT/$rel"
    if [ ! -f "$script" ]; then
        echo "[$i/$TOTAL] SKIP $rel (文件不存在)"
        continue
    fi
    printf "[%d/%d] %-65s ... " "$i" "$TOTAL" "$(basename "$rel")"
    t0=$(date +%s)
    if python "$script" >/tmp/real_demo_$$.log 2>&1; then
        t1=$(date +%s)
        elapsed=$((t1 - t0))
        # 从输出提取关键信息
        summary=$(grep -E "多数投票|答案|content|content|response|\[✓\]|\[API ERROR\]|samples|chunk" /tmp/real_demo_$$.log 2>/dev/null | head -1 | head -c 60)
        echo "OK (${elapsed}s) ${summary}"
        PASS=$((PASS+1))
    else
        t1=$(date +%s)
        elapsed=$((t1 - t0))
        echo "FAIL (${elapsed}s)"
        # 打印关键错误
        grep -E "Error|ERROR|Traceback" /tmp/real_demo_$$.log | head -2
        FAIL=$((FAIL+1))
    fi
done
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

rm -f /tmp/real_demo_$$.log

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  完成: $PASS passed, $FAIL failed (总计 $TOTAL, 耗时 ${TOTAL_ELAPSED}s)"
echo "═══════════════════════════════════════════════════════════════"

# 估算花费 (按 DeepSeek 价格: ¥1/百万 input token)
echo ""
echo "  估算成本 (按 DeepSeek 价格):"
echo "    DeepSeek:    ¥1/百万 input tokens, ¥2/百万 output"
echo "    Kimi:        ¥12/百万 tokens"
echo "    SiliconFlow: 部分模型免费 (Qwen 7B), 其他 ¥1-4/百万"
echo "    MiniMax:  按订阅计划"
echo ""
echo "  本次运行约 10-50 个 chat 调用, 估计 ¥0.01 - ¥0.10"

exit $FAIL
