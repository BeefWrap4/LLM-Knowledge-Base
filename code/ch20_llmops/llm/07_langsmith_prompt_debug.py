# ---
# chapter: 20
# topic: LLMOps与模型可观测性
# section: 20.3.4 Prompt 调试与优化
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: langsmith (mocked fallback)
# run: python 07_langsmith_prompt_debug.py
# expected_runtime: < 1s
# expected_output: Diagnostic report dict with truncated/empty/long prompt buckets
# ---
# See: ../tutorial/20_LLMOps与模型可观测性.md#2034-prompt-调试与优化-⭐⭐⭐
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


def debug_prompt_issue(project_name: str, query_pattern: str) -> dict[str, list]:
    """通过 LangSmith API 批量诊断 Prompt 问题（mocked）"""
    runs = _mock_list_runs(
        project_name=project_name,
        execution_order=1,
        filter='eq(name, "Build Prompt")',
    )

    issues: dict[str, list] = {
        "truncated_prompts": [],
        "empty_contexts": [],
        "long_prompts": [],
        "malformed_outputs": [],
    }

    for run in runs:
        prompt_text = run.outputs.get("prompt", "") if run.outputs else ""
        input_data = run.inputs or {}

        # 检测1：Prompt 是否为空或过短
        if len(prompt_text) < 50:
            issues["truncated_prompts"].append(
                {
                    "run_id": run.id,
                    "prompt_length": len(prompt_text),
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

        # 检测3：Prompt 是否过长（可能被 API 截断）
        if len(prompt_text) > 8000:
            issues["long_prompts"].append(
                {
                    "run_id": run.id,
                    "prompt_length": len(prompt_text),
                }
            )

    print("=== Prompt 诊断报告 (mocked) ===")
    print(f"总 Trace 数: {len(runs)}")
    print(f"截断/过短: {len(issues['truncated_prompts'])}")
    print(f"空上下文: {len(issues['empty_contexts'])}")
    print(f"过长 Prompt: {len(issues['long_prompts'])}")
    return issues


if __name__ == "__main__":
    issues = debug_prompt_issue("my-qa-system", "customer support")
    print(issues)
