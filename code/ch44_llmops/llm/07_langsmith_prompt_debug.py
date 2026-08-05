# ---
# chapter: 45
# topic: 大模型可观测性与 SRE
# topic_id: llmops.langsmith_prompt_debug
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langsmith (mocked fallback)
# run: python 07_langsmith_prompt_debug.py
# expected_runtime: < 1s
# expected_output: Diagnostic report with configurable short/empty/long character-count buckets
# ---
# See: ../../../45_大模型可观测性与SRE.md
# Interview hooks:
#  - 如何用 LangSmith API 批量诊断 Prompt 问题？
#  - Trace 中 run.outputs / run.inputs 分别在什么时机填充？
#  - 哪些是 Prompt 调试的常见可观测信号？


# 离线 mock：构造伪造的 run 列表
class _MockRun:
    def __init__(self, run_id: str, inputs: dict, outputs: dict):
        self.id = run_id
        self.inputs = inputs
        self.outputs = outputs


def _mock_list_runs(**kwargs):
    """伪造的 list_runs：3 个 run，分别覆盖截断/正常/过长。"""
    return [
        _MockRun(
            run_id="run_1",
            inputs={"question": "短问", "context": ""},
            outputs={"prompt": "短prompt"},
        ),
        _MockRun(
            run_id="run_2",
            inputs={"question": "正常问", "context": "丰富上下文"},
            outputs={"prompt": "x" * 200},
        ),
        _MockRun(
            run_id="run_3",
            inputs={"question": "长问", "context": "大段背景"},
            outputs={"prompt": "y" * 9000},
        ),
    ]


def debug_prompt_issue(
    project_name: str,
    *,
    min_prompt_chars: int,
    max_prompt_chars: int,
) -> dict[str, list]:
    """用可配置的字符数规则筛选 Trace；字符数不是模型 Token/截断判据。"""
    if min_prompt_chars < 0 or max_prompt_chars <= min_prompt_chars:
        raise ValueError("expected 0 <= min_prompt_chars < max_prompt_chars")
    runs = _mock_list_runs(
        project_name=project_name,
        execution_order=1,
        filter='eq(name, "Build Prompt")',
    )

    issues: dict[str, list] = {
        "short_prompts": [],
        "empty_contexts": [],
        "long_prompts_by_chars": [],
        "malformed_outputs": [],
    }

    for run in runs:
        prompt_text = run.outputs.get("prompt", "") if run.outputs else ""
        input_data = run.inputs or {}

        # 字符数只用于初筛；是否截断要读取真实 tokenizer/usage 与上下文上限。
        if len(prompt_text) < min_prompt_chars:
            issues["short_prompts"].append(
                {
                    "run_id": run.id,
                    "prompt_chars": len(prompt_text),
                    "input": input_data,
                }
            )

        # 检测2：上下文是否为空
        if not input_data.get("context"):
            issues["empty_contexts"].append(
                {
                    "run_id": run.id,
                    "question": input_data.get("question"),
                }
            )

        if len(prompt_text) > max_prompt_chars:
            issues["long_prompts_by_chars"].append(
                {
                    "run_id": run.id,
                    "prompt_chars": len(prompt_text),
                }
            )

    print("=== Prompt 诊断报告 (mocked) ===")
    print(f"总 Trace 数: {len(runs)}")
    print(f"字符数偏短: {len(issues['short_prompts'])}")
    print(f"空上下文: {len(issues['empty_contexts'])}")
    print(f"字符数偏长: {len(issues['long_prompts_by_chars'])}")
    return issues


if __name__ == "__main__":
    # 20/8000 是可复现教学筛选器，不代表模型上下文边界。
    issues = debug_prompt_issue(
        "my-qa-system",
        min_prompt_chars=20,
        max_prompt_chars=8000,
    )
    print(issues)
    print("OK")
