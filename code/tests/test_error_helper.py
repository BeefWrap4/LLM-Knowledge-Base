"""Tests for shared._error_helper unified error formatting.

Task: Wave 1 / Task 1 — provide a single exit point for RuntimeError
messages so all 8-wave refactor tasks produce consistent user-facing
output with a `[ERROR]` / `[HELP]` shape and an optional file:line
location prefix.
"""
import pytest

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


def test_raise_with_help_raises_runtime_error_by_default():
    """raise_with_help 默认抛 RuntimeError."""
    with pytest.raises(RuntimeError) as exc_info:
        from shared._error_helper import raise_with_help
        raise_with_help("test message", "test hint")
    msg = str(exc_info.value)
    assert "[ERROR]" in msg
    assert "test message" in msg
    assert "[HELP]" in msg
    assert "test hint" in msg


def test_raise_with_help_custom_exception_class():
    """raise_with_help 接受自定义 exc_class."""
    with pytest.raises(ValueError) as exc_info:
        from shared._error_helper import raise_with_help
        raise_with_help("custom msg", "custom hint", exc_class=ValueError)
    assert "custom msg" in str(exc_info.value)


def test_format_error_full_string_golden():
    """golden test: 锁死完整输出格式."""
    msg = format_error(
        "缺权重",
        "运行 make download-models",
        file_path="ch25/10_vllm_async_engine.py",
        line=42,
    )
    assert msg == (
        "[ERROR] ch25/10_vllm_async_engine.py:42  缺权重\n"
        "[HELP]  运行 make download-models\n"
        "[HELP]  或 `export LLM_MOCK=1` 用 mock 跑 (仅 CI/离线)"
    )
