"""Tests for shared._error_helper unified error formatting.

Task: Wave 1 / Task 1 — provide a single exit point for RuntimeError
messages so all 8-wave refactor tasks produce consistent user-facing
output with a `[ERROR]` / `[HELP]` shape and an optional file:line
location prefix.
"""
from shared._error_helper import format_error


def test_format_error_basic():
    msg = format_error("缺 API Key", "运行 make llm-doctor")
    assert "[ERROR]" in msg
    assert "缺 API Key" in msg
    assert "[HELP]" in msg
    assert "make llm-doctor" in msg


def test_format_error_with_file():
    msg = format_error(
        "缺权重",
        "运行 make download-models",
        file_path="ch25/10_vllm_async_engine.py",
        line=42,
    )
    assert "ch25/10_vllm_async_engine.py:42" in msg
