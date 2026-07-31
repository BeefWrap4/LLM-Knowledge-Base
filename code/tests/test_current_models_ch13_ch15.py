"""Regression checks for current model defaults in chapters 13-15."""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = [
    REPO_ROOT / "13_Prompt_Engineering.md",
    REPO_ROOT / "14_RAG检索增强生成.md",
    REPO_ROOT / "15_Agent智能体开发.md",
]
CODE_DIRS = [
    REPO_ROOT / "code" / "ch13_prompt_engineering",
    REPO_ROOT / "code" / "ch14_rag",
    REPO_ROOT / "code" / "ch15_agent",
]
RETIRED_OPENAI_MODEL = re.compile(
    r"(?<![\w.-])(?:gpt-4(?:-turbo)?|gpt-3\.5-turbo|gpt-4o(?:-mini)?|"
    r"o1(?:-mini|-preview)?|o3-mini)(?![\w.-])",
    re.IGNORECASE,
)


def _scoped_text_files() -> list[Path]:
    files = list(CHAPTERS)
    for directory in CODE_DIRS:
        files.extend(directory.rglob("*.py"))
        files.extend(directory.rglob("*.md"))
    return files


def _load_contextual_retrieval():
    path = REPO_ROOT / "code" / "ch14_rag" / "llm" / "20_contextual_retrieval.py"
    spec = importlib.util.spec_from_file_location("_test_contextual_retrieval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scoped_content_has_no_retired_openai_model_literal():
    findings = []
    for path in _scoped_text_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if match := RETIRED_OPENAI_MODEL.search(line):
                findings.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {match.group(0)}")

    assert findings == []


@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda path: path.name)
def test_chapters_use_configurable_current_openai_default(chapter: Path):
    text = chapter.read_text(encoding="utf-8")
    assert 'os.getenv("OPENAI_MODEL", "gpt-5.6")' in text


def test_contextual_retrieval_defaults_to_offline_even_with_client(monkeypatch):
    module = _load_contextual_retrieval()

    class FailMessages:
        @staticmethod
        def create(**_kwargs):
            raise AssertionError("default mock path must not call the client")

    monkeypatch.delenv("LLM_MOCK", raising=False)
    client = SimpleNamespace(messages=FailMessages())
    result = module.add_context_to_chunk("年假 15 天。", "公司人事手册。", client=client)

    assert result.startswith("[上下文：本文档主题:")


def test_contextual_retrieval_real_mode_requires_explicit_client(monkeypatch):
    module = _load_contextual_retrieval()
    monkeypatch.setenv("LLM_MOCK", "0")

    with pytest.raises(RuntimeError, match="Anthropic client"):
        module.add_context_to_chunk("年假 15 天。", "公司人事手册。")
