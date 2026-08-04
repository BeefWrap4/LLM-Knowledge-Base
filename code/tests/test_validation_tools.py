"""Regression tests for fail-closed repository validation."""

import sys
from pathlib import Path

import pytest

from scripts import normalize_chapter_narrative, run_all_examples, sync_links, verify_all


def test_tutorial_section_parser_supports_four_levels() -> None:
    match = sync_links.TUTORIAL_SECTION_RE.match("### 24.6.5.1 主流推理引擎")
    assert match is not None
    number, title = match.groups()
    assert number == "24.6.5.1"
    assert title == "主流推理引擎"


def test_code_section_parser_supports_four_levels() -> None:
    match = sync_links.CODE_SECTION_RE.match("# section: 24.6.5.1 K8s adapter")
    assert match is not None
    number, title = match.groups()
    assert number == "24.6.5.1"
    assert title == "K8s adapter"


def test_llm_runner_defaults_to_mock(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "example.py"
    script.write_text(
        "import os\nassert os.environ.get('LLM_MOCK') == '1'\nprint('OK')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_all_examples, "CODE", tmp_path)
    monkeypatch.setattr(run_all_examples, "PY", sys.executable)
    monkeypatch.delenv("LLM_MOCK", raising=False)

    rel, passed, output, _ = run_all_examples.run_one(script, timeout=10, tier="llm")

    assert rel == "example.py"
    assert passed, output


def test_llm_runner_overrides_parent_real_mode(monkeypatch, tmp_path: Path) -> None:
    """批量验收不能继承父进程的真实 API 开关。"""
    script = tmp_path / "example.py"
    script.write_text(
        "import os\nassert os.environ.get('LLM_MOCK') == '1'\nprint('OK')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_all_examples, "CODE", tmp_path)
    monkeypatch.setattr(run_all_examples, "PY", sys.executable)
    monkeypatch.setenv("LLM_MOCK", "0")

    _, passed, output, _ = run_all_examples.run_one(script, timeout=10, tier="llm")

    assert passed, output


def test_runner_preserves_skip_marker_before_truncation(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "example.py"
    script.write_text(
        "print('[SKIP] ' + 'requirement-' * 30)\nprint('OK')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_all_examples, "CODE", tmp_path)
    monkeypatch.setattr(run_all_examples, "PY", sys.executable)

    _, passed, output, _ = run_all_examples.run_one(script, timeout=10, tier="gpu")

    assert passed, output
    assert "[SKIP]" in output


def test_runner_rejects_silent_success_without_ok(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "example.py"
    script.write_text("print('Skipping real mode')\n", encoding="utf-8")
    monkeypatch.setattr(run_all_examples, "CODE", tmp_path)
    monkeypatch.setattr(run_all_examples, "PY", sys.executable)

    _, passed, output, _ = run_all_examples.run_one(script, timeout=10, tier="gpu")

    assert not passed
    assert "[MISSING OK MARKER]" in output


def test_gpu_runner_mock_flag_is_explicitly_disableable(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "example.py"
    script.write_text(
        "import sys\nprint('MOCK=' + str('--mock' in sys.argv))\nprint('OK')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_all_examples, "CODE", tmp_path)
    monkeypatch.setattr(run_all_examples, "PY", sys.executable)

    _, default_passed, default_output, _ = run_all_examples.run_one(script, timeout=10, tier="gpu")
    _, real_passed, real_output, _ = run_all_examples.run_one(script, timeout=10, tier="gpu", real_gpu=True)

    assert default_passed and "MOCK=True" in default_output
    assert real_passed and "MOCK=False" in real_output


@pytest.mark.parametrize(
    "argv",
    [
        ["run_all_examples.py", "--tier", "gpu", "--real-gpu", "--parallel", "1"],
        [
            "run_all_examples.py",
            "--tier",
            "gpu",
            "--real-gpu",
            "--chapter",
            "ch21",
            "--parallel",
            "2",
        ],
    ],
)
def test_real_gpu_runner_requires_narrow_serial_scope(monkeypatch, argv) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        run_all_examples.main()

    assert exc_info.value.code == 2


def test_current_repository_snapshot() -> None:
    assert len(verify_all.canonical_chapters()) == verify_all.EXPECTED_CHAPTERS
    examples = list(verify_all.CODE.glob("ch[0-9][0-9]_*/*/*.py"))
    assert len(examples) == verify_all.EXPECTED_EXAMPLES
    for tier in ("core", "llm", "gpu"):
        assert (verify_all.CODE / f"requirements-{tier}.ci.lock").is_file()


def test_current_mermaid_blocks_are_obsidian_safe() -> None:
    total, failures = verify_all.inspect_mermaid_blocks()

    assert total == 261
    assert failures == []


def test_current_markdown_documents_are_obsidian_safe() -> None:
    total, failures = verify_all.inspect_markdown_rendering()

    assert total == 101
    assert failures == []


def test_current_chapter_narratives_follow_learning_contract() -> None:
    total, failures = verify_all.inspect_chapter_narratives()

    assert total == 40
    assert failures == []


def test_chapter_normalizer_is_idempotent() -> None:
    for chapter in verify_all.canonical_chapters():
        assert normalize_chapter_narrative.normalize_chapter(chapter) == chapter.read_text(encoding="utf-8")


def test_chapter_narrative_gate_rejects_incomplete_learning_contract(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "01_Test.md").write_text(
        "---\n"
        "chapter: 1\n"
        "updated: 2026-08-04T00:00:00.000Z\n"
        "---\n"
        "# 第 1 章 Test\n\n"
        "> [!abstract] 本章导航\n"
        "> **学习目标**：\n"
        "> - 解释机制。\n"
        "> - 完成实现。\n\n"
        "## 1.2 跳号正文\n\n"
        "### 9.9.1 错误父级\n\n"
        "正文。\n\n"
        "## 📖 一手参考资料\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    total, failures = verify_all.inspect_chapter_narratives()

    assert total == 1
    assert any("2 learning objectives; expected 3" in failure for failure in failures)
    assert any("has no chapter introduction" in failure for failure in failures)
    assert any("main H2 numbering is not contiguous from 1" in failure for failure in failures)
    assert any("numbered descendant heading does not inherit" in failure for failure in failures)
    assert any("fixed ending sections are missing" in failure for failure in failures)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("# Title\n```python\nprint('x')\n", "unclosed Markdown fence"),
        ("# Title\n$$\nx + y\n", "unclosed display-math block"),
        ("---\ntitle: Test\n", "unclosed YAML frontmatter"),
        ("# Title\n<!-- hidden\n", "unclosed HTML comment"),
        ("# Title\n<div>\n", "unclosed <div> block"),
        ("# Title\n[[missing\n", "unbalanced Obsidian WikiLink delimiters"),
        ("# Title\n> [!NOTE\n", "malformed Obsidian callout header"),
        ("# Title\nAn unclosed $x expression\n", "unbalanced inline-math '$' delimiter"),
        ("# Title\n### Skipped level\n", "heading level jumps from H1 to H3"),
    ],
)
def test_markdown_render_gate_rejects_unclosed_structures(
    monkeypatch, tmp_path: Path, body: str, message: str
) -> None:
    (tmp_path / "bad.md").write_text(body, encoding="utf-8")
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    total, failures = verify_all.inspect_markdown_rendering()

    assert total == 1
    assert any(message in failure for failure in failures)


def test_markdown_render_gate_rejects_table_column_mismatch(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text(
        "| A | B |\n|---|---|\n| one | two | three |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    assert verify_all.inspect_markdown_rendering() == (
        1,
        ["bad.md:3 table row has 3 cells; expected 2"],
    )


def test_markdown_render_gate_rejects_math_crossing_table_cells(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text(
        "| Kernel | Formula start | Midpoint | Formula end | Use |\n"
        "|---|---|---|---|---|\n"
        "| RBF | $K(x,y)=\\exp(-\\gamma | x-y | ^2)$ | general |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    _, failures = verify_all.inspect_markdown_rendering()

    assert failures == ["bad.md:3 inline math crosses table cell boundaries at columns 2, 4"]


def test_markdown_render_gate_rejects_unescaped_underscore_inside_latex_text(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "bad.md").write_text(
        "$$\n\\text{batch_size} = 1\n$$\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    assert verify_all.inspect_markdown_rendering() == (
        1,
        [
            "bad.md:2 unescaped underscore inside LaTeX \\text{...}; "
            r"escape it as \_ or use symbolic subscripts"
        ],
    )


def test_markdown_render_gate_rejects_multiple_empty_table_headers(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text(
        "| Kernel | Formula | | | Use |\n|---|---|---|---|---|\n| RBF | expression | x | y | general |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    assert verify_all.inspect_markdown_rendering() == (
        1,
        ["bad.md:1 table header has multiple empty cells at columns 3, 4; remove accidental columns"],
    )


def test_markdown_render_gate_rejects_nested_triple_fence_in_markdown_example(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "bad.md").write_text(
        "```markdown\n# Example\n```python\nprint('x')\n```\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    _, failures = verify_all.inspect_markdown_rendering()

    assert any("nested fence can terminate Markdown example" in failure for failure in failures)


@pytest.mark.parametrize(
    ("link", "message"),
    [
        ("[missing](missing.md)", "unresolved local Markdown target: missing.md"),
        ("[[missing]]", "unresolved Obsidian WikiLink target: missing"),
    ],
)
def test_markdown_render_gate_rejects_broken_local_links(
    monkeypatch, tmp_path: Path, link: str, message: str
) -> None:
    (tmp_path / "bad.md").write_text(f"# Title\n{link}\n", encoding="utf-8")
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    total, failures = verify_all.inspect_markdown_rendering()

    assert total == 1
    assert failures == [f"bad.md:2 {message}"]


def test_markdown_render_gate_rejects_symlink_dependent_link(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text("[chapter](alias/chapter.md)\n", encoding="utf-8")
    monkeypatch.setattr(verify_all, "REPO", tmp_path)
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path.name == "alias" or original_is_symlink(path),
    )

    _, failures = verify_all.inspect_markdown_rendering()

    assert len(failures) == 1
    assert "local Markdown target traverses symlink" in failures[0]


def test_markdown_render_gate_accepts_supported_obsidian_syntax(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "target.md").write_text("# Target\n", encoding="utf-8")
    (tmp_path / "good.md").write_text(
        "---\n"
        "title: Good\n"
        "---\n"
        "# Good\n"
        "> [!NOTE]+ Expanded\n"
        "> Body\n\n"
        "| Syntax | Meaning |\n"
        "|---|---|\n"
        r"| `a | b` | logical or \| pipe |"
        "\n\n"
        "| Metric | Formula |\n"
        "|---|---|\n"
        "| Norm | $\\lVert x \\rVert_2$ |\n\n"
        "[[target#Target|alias]] and [target](target.md#target)\n"
        "<details><summary>More</summary>Text</details>\n"
        "$$\n\\text{batch\\_size} + y\n$$\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    assert verify_all.inspect_markdown_rendering() == (2, [])


def test_mermaid_gate_rejects_unclosed_fence(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text("```mermaid\ntree\nroot --> leaf\n", encoding="utf-8")
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    total, failures = verify_all.inspect_mermaid_blocks()

    assert total == 0
    assert failures == ["bad.md:1 unclosed Mermaid fence"]


def test_mermaid_gate_rejects_unknown_diagram(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text(
        "```mermaid\ntree\nroot --> leaf\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    total, failures = verify_all.inspect_mermaid_blocks()

    assert total == 1
    assert failures == ["bad.md:2 unsupported Mermaid diagram: tree"]


def test_mermaid_gate_rejects_html_break_in_timeline(monkeypatch, tmp_path: Path) -> None:
    """Obsidian 会显示 timeline 条目中的 <br/>，并把长文本压进窄列。"""
    (tmp_path / "bad.md").write_text(
        "```mermaid\ntimeline\n    title 演进路线\n    2026 : RAG-as-a-Tool<br/>多模态 RAG + 端云协同\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    total, failures = verify_all.inspect_mermaid_blocks()

    assert total == 1
    assert failures == [
        "bad.md:4 unsupported Mermaid HTML line break in timeline; "
        "use a flowchart for multi-line stage descriptions"
    ]


def test_mermaid_gate_rejects_react_messages_routed_through_user(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text(
        "```mermaid\n"
        "sequenceDiagram\n"
        "participant U as User\n"
        "participant A as LLM Agent\n"
        "participant T as External Tool\n"
        "A-->>U: Action: search(query)\n"
        "U->>T: 执行搜索\n"
        "T-->>U: Observation: result\n"
        "```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    total, failures = verify_all.inspect_mermaid_blocks()

    assert total == 1
    assert failures == [
        "bad.md:6 ReAct Action is routed to the user; route it to the tool/runtime",
        "bad.md:7 user is acting as the tool executor; route the call through the agent/runtime",
        "bad.md:8 ReAct Observation is routed to the user; route it to the agent/runtime",
    ]


def test_mermaid_gate_accepts_react_agent_tool_routing(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text(
        "```mermaid\n"
        "sequenceDiagram\n"
        "participant U as 用户\n"
        "participant A as Agent (LLM)\n"
        "participant T as 工具/API\n"
        "U->>A: 提问\n"
        "A->>T: Action: search(query)\n"
        "T-->>A: Observation: result\n"
        "A-->>U: Final Answer: answer\n"
        "```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    assert verify_all.inspect_mermaid_blocks() == (1, [])


@pytest.mark.parametrize(
    "label",
    [
        "NODE[全[MASK]序列]",
        "NODE[Z ∈ R^{N×d}]",
        "NODE[epsilon(z_t, t)]",
        'NODE["块1 [16]")',
    ],
)
def test_mermaid_gate_rejects_parser_sensitive_unquoted_labels(
    monkeypatch, tmp_path: Path, label: str
) -> None:
    (tmp_path / "bad.md").write_text(
        f"```mermaid\nflowchart TD\n{label}\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    total, failures = verify_all.inspect_mermaid_blocks()

    assert total == 1
    assert len(failures) == 1
    assert "quote the label" in failures[0]


@pytest.mark.parametrize(
    "label",
    [
        'NODE["1. first step"]',
        'A -->|"2. call"| B',
        'NODE["+"]',
        "A -->|+| B",
        "A -->|> 0| B",
    ],
)
def test_mermaid_gate_rejects_unsupported_markdown_at_label_start(
    monkeypatch, tmp_path: Path, label: str
) -> None:
    (tmp_path / "bad.md").write_text(
        f"```mermaid\nflowchart TD\n{label}\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    total, failures = verify_all.inspect_mermaid_blocks()

    assert total == 1
    assert len(failures) == 1
    assert "unsupported Mermaid Markdown" in failures[0]


def test_mermaid_gate_accepts_quoted_labels_and_cylinder_shape(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text(
        "```mermaid\n"
        "flowchart TD\n"
        'MASK["全 [MASK] 序列"] --> MATH["epsilon(z_t, t) ∈ R^{N×d}"]\n'
        "MATH --> CACHE[(Redis Cache)]\n"
        'CACHE --> STEP["步骤 1：读取"]\n'
        'STEP --> ADD["相加（+）"]\n'
        "ADD -->|大于 0| DONE\n"
        "```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    assert verify_all.inspect_mermaid_blocks() == (1, [])


def test_mermaid_gate_accepts_state_diagram_terminal_nodes(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "state.md").write_text(
        "```mermaid\nstateDiagram-v2\n[*] --> Plan\nPlan --> [*]\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)

    assert verify_all.inspect_mermaid_blocks() == (1, [])


def test_python_reference_gate_detects_missing_file(monkeypatch, tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (tmp_path / "README.md").write_text(
        "python ch15_agent/llm/01_missing.py\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)
    monkeypatch.setattr(verify_all, "CODE", code)

    assert verify_all.find_broken_python_references() == [("README.md", 1, "ch15_agent/llm/01_missing.py")]


def test_python_reference_gate_accepts_existing_code_prefix(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "code/ch15_agent/llm/01_agent.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('OK')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "python code/ch15_agent/llm/01_agent.py\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "REPO", tmp_path)
    monkeypatch.setattr(verify_all, "CODE", tmp_path / "code")

    assert verify_all.find_broken_python_references() == []


def test_gpu_mock_contract_rejects_unmarked_script(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "ch25_demo/gpu/01_demo.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('OK')\n", encoding="utf-8")
    monkeypatch.setattr(verify_all, "CODE", tmp_path)

    assert "missing skip_if_mock" in verify_all.find_gpu_mock_contract_failures()[0]


def test_gpu_mock_safe_metadata_rejects_network_like_call(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "ch25_demo/gpu/01_demo.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "# ---\n# tier: gpu\n# mock_safe: true\n# ---\n"
        "from transformers import AutoModel\n"
        "AutoModel.from_pretrained('remote/model')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_all, "CODE", tmp_path)

    assert "from_pretrained" in verify_all.find_gpu_mock_contract_failures()[0]


def test_examples_do_not_use_builtin_eval_or_exec() -> None:
    assert verify_all.find_dynamic_execution_failures() == []


def test_dynamic_execution_gate_rejects_eval(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "ch15_demo/llm/01_demo.py"
    script.parent.mkdir(parents=True)
    script.write_text("result = eval(model_output)\n", encoding="utf-8")
    monkeypatch.setattr(verify_all, "CODE", tmp_path)

    assert "builtin eval() is forbidden" in verify_all.find_dynamic_execution_failures()[0]
