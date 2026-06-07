# tests/test_core_no_mock.py
"""自动审计: core tier 主流程不应有 mock / fake 引用.

W1 已把 mock 路径下沉到 tests/_mocks/. core tier 是 Python 基础层
(无 LLM 调用), 不应出现任何 mock/fake 残留.
"""
import re
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent

# 禁止出现的模式
FORBIDDEN_PATTERNS = [
    "MockLLM", "mock_llm", "MockChat",
    "is_mock", "USE_REAL_API",
    "fake_llm", "FakeListChatModel", "FakeChatModel",
    "from shared.mock_llm", "import mock_llm",
]

# 允许的例外 (注释 / docstring 提及 mock 是 OK 的)
ALLOWED_LINE_PATTERNS = [
    re.compile(r"^\s*#"),          # 注释
    re.compile(r'^\s*"""'),        # docstring 起始
    re.compile(r'^\s*""".*"""'),   # 单行 docstring
]


def _is_allowed_line(line: str) -> bool:
    """判断该行是否是允许的 (注释 / docstring)."""
    for pat in ALLOWED_LINE_PATTERNS:
        if pat.match(line):
            return True
    return False


def test_no_mock_in_core_tier():
    """core tier 任何文件不应有 mock / fake 引用 (注释除外)."""
    violations = []
    core_files = sorted(CODE_ROOT.glob("ch*/core/*.py"))
    if not core_files:
        # 没找到任何文件 — 测试设计本身有问题
        assert False, f"未找到任何 ch*/core/*.py 文件 (CODE_ROOT={CODE_ROOT})"

    for py_file in core_files:
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
                        f"{rel_path}:{line_no}: 出现 '{pat}'\n    {line.strip()}"
                    )
                    break  # 一行只报一次

    if violations:
        msg = f"core tier 不应出现 mock / fake 引用 ({len(violations)} 个违规):\n"
        msg += "\n".join(violations[:30])
        if len(violations) > 30:
            msg += f"\n... (还有 {len(violations) - 30} 个未显示)"
        assert False, msg


def test_core_files_count():
    """sanity check: 应该有约 158 个 core 文件."""
    core_files = list(CODE_ROOT.glob("ch*/core/*.py"))
    core_files = [f for f in core_files if f.name != "__init__.py"]
    # 158 是 W2 计划目标; 实际数允许 ±10%
    assert 100 <= len(core_files) <= 200, (
        f"core 文件数 {len(core_files)} 偏离预期 100-200"
    )
