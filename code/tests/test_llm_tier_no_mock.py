# tests/test_llm_tier_no_mock.py
"""自动审计: llm tier 主流程不应有 mock / fake 引用 (除 LLM_MOCK 单行开关外).

W1 已下沉 mock 到 tests/_mocks/. W3 目标: 移除 llm tier 主流程的所有 mock 分支.
任何 PR 引入新的 mock 引用, CI 立即 fail.
"""
import re
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent

# 允许的"单行开关" — 仅当 LLM_MOCK 单行检查时, 允许特定模式
ALLOWED_LINE_PATTERN = re.compile(r'os\.environ\.get\(["\']LLM_MOCK["\']\)')

# 禁止的模式 (会出现在 LLM tier 主流程)
FORBIDDEN_PATTERNS = [
    "is_mock",
    "USE_REAL_API",
    'provider == "mock"',
    'provider=="mock"',
    "MockLLM()",
    "fake_llm",
    "FakeListChatModel",
    "FakeChatModel",
    "from shared.mock_llm",
    "import mock_llm",
]

# 允许的"已知文本" (注释 / docstring 提及 mock 但实际未使用)
ALLOWED_LINE_PATTERNS = [
    re.compile(r"^\s*#"),
    re.compile(r'^\s*"""'),
    re.compile(r'^\s*""".*"""'),
]


def _is_allowed_line(line: str) -> bool:
    """判断该行是否是允许的 (注释 / docstring / LLM_MOCK 单行开关)."""
    for pat in ALLOWED_LINE_PATTERNS:
        if pat.match(line):
            return True
    if ALLOWED_LINE_PATTERN.search(line):
        return True
    return False


def test_no_mock_in_llm_tier():
    """llm tier 主流程不应有 mock 引用 (注释 + LLM_MOCK 单行开关除外)."""
    violations = []
    llm_files = sorted(CODE_ROOT.glob("ch*/llm/*.py"))

    for py_file in llm_files:
        if py_file.name == "__init__.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line_no, line in enumerate(content.split("\n"), 1):
            if _is_allowed_line(line):
                continue
            for pat in FORBIDDEN_PATTERNS:
                if pat in line:
                    rel_path = py_file.relative_to(CODE_ROOT)
                    violations.append(
                        f"{rel_path}:{line_no}: '{pat}'\n    {line.strip()}"
                    )
                    break  # 一行只报一次

    if violations:
        msg = f"llm tier 不应出现 mock 引用 ({len(violations)} 个违规):\n"
        msg += "\n".join(violations[:50])
        if len(violations) > 50:
            msg += f"\n... (还有 {len(violations) - 50} 个未显示)"
        assert False, msg


def test_llm_files_count():
    """sanity check: 应该有约 199 个 llm 文件 (W3 计划目标)."""
    llm_files = list(CODE_ROOT.glob("ch*/llm/*.py"))
    llm_files = [f for f in llm_files if f.name != "__init__.py"]
    assert 100 <= len(llm_files) <= 300, (
        f"llm 文件数 {len(llm_files)} 偏离预期 100-300"
    )
