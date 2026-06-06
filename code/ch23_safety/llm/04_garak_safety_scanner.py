# ---
# chapter: 23
# topic: AI安全与伦理
# section: 23.3.4 Garak安全评估工具
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: garak (optional, only for actual execution)
# run: python 04_garak_safety_scanner.py
# expected_runtime: <1s
# expected_output: 演示Garak CLI调用信息 + 2026新增探测器列表 + "OK"
# ---
# See: ../tutorial/23_AI安全与伦理.md#2334-开源越狱检测工具
# Interview hooks:
#   1. Garak的核心功能是什么？它支持哪些类型的探测器（Probes）？
#   2. 如何将Garak集成到CI/CD流程中实现安全门禁？
#   3. 2026年Garak新增了哪些针对Agent场景的探测器？
"""
使用Garak进行LLM安全评估

面试中谈论Garak时应该了解的要点：
1. Garak是什么：NVIDIA开源的LLM漏洞扫描器
2. 支持的探测器（Probes）类型
3. 如何集成到CI/CD流程中
"""


# Garak CLI 使用示例（面试中口述即可）
GARAK_CLI_EXAMPLES = """
# 安装
pip install garak

# 对目标模型进行完整安全扫描
garak --model_type huggingface \\
      --model_name meta-llama/Llama-3-8B-Instruct \\
      --probes dan,encoding,knownbadsignatures,toxicity

# 只检测越狱漏洞
garak --model_type openai \\
      --model_name gpt-4 \\
      --probes jailbreak

# 生成HTML报告
garak --model_type huggingface \\
      --model_name meta-llama/Llama-3-8B-Instruct \\
      --report_prefix my_audit \\
      --report_format html
"""

# 🆕 2026年Garak新增探测器类型
GARAK_NEW_PROBES_2026 = {
    "many-shot": "检测Many-shot越狱攻击",
    "multilingual_jailbreak": "多语言越狱检测",
    "indirect_injection": "间接注入检测（RAG场景）",
    "agent_tool_abuse": "Agent工具滥用检测",
    "chain_of_thought_hijack": "思维链劫持检测",
}


# ========== Python包装函数（程序化调用Garak） ==========
def run_garak_scan(
    model_type: str,
    model_name: str,
    probes: str = "dan,encoding,toxicity",
    report_format: str = "html",
    report_prefix: str = "audit"
) -> dict:
    """程序化调用Garak执行安全扫描

    Args:
        model_type: 模型类型（huggingface/openai/...）
        model_name: 模型名称或ID
        probes: 探测器列表（逗号分隔）
        report_format: 报告格式（html/json）
        report_prefix: 报告文件前缀

    Returns:
        扫描结果摘要（mock-mode下返回固定结构）
    """
    # mock-mode fallback: 实际调用应使用subprocess执行garak CLI
    # 真实实现示例：
    # import subprocess
    # cmd = [
    #     "garak",
    #     "--model_type", model_type,
    #     "--model_name", model_name,
    #     "--probes", probes,
    #     "--report_format", report_format,
    #     "--report_prefix", report_prefix,
    # ]
    # result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "status": "mock",
        "model": model_name,
        "probes": probes.split(","),
        "report": f"{report_prefix}.{report_format}",
        "note": "实际部署请通过subprocess调用garak CLI",
    }


if __name__ == "__main__":
    print("=== Garak 安全评估工具演示 ===")
    print("\nGarak CLI调用示例：")
    print(GARAK_CLI_EXAMPLES)

    print("🆕 2026年新增探测器：")
    for probe, desc in GARAK_NEW_PROBES_2026.items():
        print(f"  - {probe}: {desc}")

    # mock扫描
    result = run_garak_scan(
        model_type="huggingface",
        model_name="meta-llama/Llama-3-8B-Instruct",
        probes="dan,encoding",
    )
    print(f"\n模拟扫描结果: {result}")
    print("OK")
