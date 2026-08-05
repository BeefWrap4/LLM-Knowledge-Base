"""Current-model and framework-API regressions for Ch18."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_ROOT.parent
CHAPTER = REPO_ROOT / "27_LLM框架与平台选型.md"
EXAMPLE_ROOT = CODE_ROOT / "ch27_llm_frameworks"
OFFLINE_ENTRYPOINTS = [
    "01_langchain_basic_chain.py",
    "02_llmchain_basic.py",
    "03_sequential_chain.py",
    "04_router_chain.py",
    "05_conversation_buffer_memory.py",
    "06_conversation_summary_memory.py",
    "07_summary_buffer_memory.py",
    "09_chatbot_with_memory.py",
]
OFFLINE_LLAMA_INDEX_ENTRYPOINTS = [
    "13_llamaindex_vectorstore_index.py",
    "14_llamaindex_summary_index.py",
    "15_llamaindex_tree_index.py",
    "16_llamaindex_keyword_index.py",
    "17_llamaindex_query_chat_engine.py",
    "18_llamaindex_enterprise_qa.py",
]
RETIRED_IDS = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4",
    "gpt-3.5",
    "claude-3",
)


def test_active_ch18_examples_do_not_pin_retired_model_ids():
    paths = [CHAPTER, *EXAMPLE_ROOT.rglob("*.py"), *EXAMPLE_ROOT.rglob("*.md")]
    for path in paths:
        source = path.read_text(encoding="utf-8").lower()
        assert not any(model_id in source for model_id in RETIRED_IDS), path
        assert "sk-dummy-for-import-only" not in source, path


def test_ch18_uses_current_framework_api_boundaries():
    chapter = CHAPTER.read_text(encoding="utf-8")
    pydantic_example = (EXAMPLE_ROOT / "llm" / "26_pydantic_ai_research_agent.py").read_text(
        encoding="utf-8"
    )
    strands_example = (EXAMPLE_ROOT / "llm" / "27_strands_agents_demo.py").read_text(
        encoding="utf-8"
    )
    haystack_example = (EXAMPLE_ROOT / "llm" / "30_haystack_rag_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "output_type=ResearchReport" in chapter
    assert "result_type=ResearchReport" not in chapter
    assert "output_type=ResearchReport" in pydantic_example
    assert "BEDROCK_MODEL_ID" in strands_example
    assert "deploy(pipe" not in chapter
    assert "deploy(pipe" not in haystack_example
    assert "SQLiteSession" in chapter
    assert "session_id=" not in chapter


@pytest.mark.parametrize("filename", OFFLINE_ENTRYPOINTS)
def test_langchain_entrypoints_are_offline_with_fake_key(filename: str):
    script = EXAMPLE_ROOT / "llm" / filename
    env = os.environ.copy()
    env["LLM_MOCK"] = "1"
    env["OPENAI_API_KEY"] = "fake-key-that-must-not-be-used"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "[SKIP]" in result.stdout
    assert "OK" in result.stdout


@pytest.mark.parametrize("filename", OFFLINE_LLAMA_INDEX_ENTRYPOINTS)
def test_llama_index_entrypoints_default_to_offline(filename: str):
    script = EXAMPLE_ROOT / "llm" / filename
    env = os.environ.copy()
    env.pop("LLM_MOCK", None)
    env["OPENAI_API_KEY"] = "fake-key-that-must-not-be-used"
    env["DEEPSEEK_API_KEY"] = "fake-key-that-must-not-be-used"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "[offline]" in result.stdout
    assert "OK" in result.stdout
