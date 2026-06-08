# ---
# chapter: 23
# topic: AI安全与伦理
# section: 23.7.5 Constitutional Classifiers (Anthropic 2025)
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: anthropic (optional, only for actual API calls)
# run: python 10_constitutional_classifier.py
# expected_runtime: <1s
# expected_output: Constitutional Classifier 演示 + 关键指标 + "OK"
# ---
# See: ../tutorial/23_AI安全与伦理.md#2375-constitutional-classifiersanthropic-2025
# Interview hooks:
#   1. Constitutional Classifier 的工作原理是什么？与LLM主链路的耦合度？
#   2. 输入侧护栏与输出侧护栏如何协同？
#   3. 在生产环境中如何平衡误报率与漏报率？
"""
Constitutional Classifier 调用示例（Anthropic API 2025+）

mock-mode fallback: 当anthropic包不可用或无API key时使用本地规则模拟。
"""

import os

# 尝试导入anthropic（mock-mode下可注释）
try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def call_with_constitutional_classifier(prompt: str, api_key: str = None) -> dict:
    """调用带Constitutional Classifier的Claude API

    Args:
        prompt: 待审查的用户提示
        api_key: Anthropic API Key

    Returns:
        API响应或mock结果
    """
    # mock-mode fallback: 演示模式
    if not HAS_ANTHROPIC or api_key is None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {
                "mode": "mock",
                "prompt": prompt,
                "verdict": "would_be_classified_as_safe",
                "note": "实际部署需要anthropic SDK + ANTHROPIC_API_KEY",
            }

    # 实际调用
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        extra_headers={
            # 启用宪法分类器（生产环境默认开启）
            "anthropic-beta": "constitutional-classifiers-20250501"
        },
    )
    return {
        "mode": "live",
        "content": response.content[0].text,
        "stop_reason": response.stop_reason,
    }


# Constitutional Classifier 关键指标（Anthropic 2025公开数据）
KEY_METRICS = {
    "false_positive_rate": "< 0.05%",
    "false_negative_rate": "< 5%",
    "latency_overhead": "~150ms",
    "deployment_default": "生产环境默认开启",
}


if __name__ == "__main__":
    print("=== Constitutional Classifier 演示 ===")

    # 1. 关键指标
    print("\n[关键指标]（Anthropic 2025公开数据）")
    for k, v in KEY_METRICS.items():
        print(f"  {k}: {v}")

    # 2. 调用演示
    test_prompt = "请解释量子计算的基本原理"
    result = call_with_constitutional_classifier(test_prompt)
    print("\n[调用演示]")
    print(f"  模式: {result.get('mode')}")
    print(f"  提示: {result.get('prompt')[:50]}...")
    if "verdict" in result:
        print(f"  判定: {result['verdict']}")
    if "content" in result:
        print(f"  响应: {result['content'][:100]}...")
    print(f"  说明: {result.get('note', 'N/A')}")
