# ---
# code/tests/test_pilots.py
# Smoke tests for Wave 0 pilot examples
# ---
"""
每个 pilot 例子必须能 `python file.py` 跑通且输出 "OK"。
"""
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run_example(rel_path: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """运行一个 example, 返回 CompletedProcess."""
    script = CODE_ROOT / rel_path
    if not script.is_file():
        pytest.skip(f"Example not found: {rel_path}")
    return subprocess.run(
        [PY, str(script)],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(CODE_ROOT),
    )


@pytest.mark.core
def test_ch01_list_dict_basics():
    """Ch01: 列表字典基础 + 4 种去重."""
    result = _run_example("ch01_python_basics/core/01_list_dict_basics.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
def test_ch02_is_vs_equals():
    """Ch02: is vs == 区别 + 小整数缓存."""
    result = _run_example("ch02_mutability/core/01_is_vs_equals.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
def test_ch03_singleton():
    """Ch03: 单例模式三种实现."""
    result = _run_example("ch03_oop/core/01_singleton.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
@pytest.mark.skipif(
    subprocess.run([PY, "-c", "import torch"], capture_output=True).returncode != 0,
    reason="torch not installed (skip if no GPU tier)"
)
def test_ch12_scaled_dot_product_attention():
    """Ch12: Scaled Dot-Product Attention."""
    result = _run_example("ch12_transformer_architecture/core/01_scaled_dot_product_attention.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
@pytest.mark.skipif(
    subprocess.run([PY, "-c", "import torch"], capture_output=True).returncode != 0,
    reason="torch not installed"
)
def test_ch12_multi_head_attention():
    """Ch12: Multi-Head Attention."""
    result = _run_example("ch12_transformer_architecture/core/02_multi_head_attention.py", timeout=60)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout


@pytest.mark.llm
def test_ch18_langchain_chain_mock():
    """Ch18: LangChain basic chain (mock 模式, 不需 API key)."""
    result = _run_example("ch18_llm_frameworks/llm/01_langchain_basic_chain.py", timeout=30)
    # 允许在没装 langchain 时 skip
    if "ModuleNotFoundError" in result.stderr and "langchain" in result.stderr:
        pytest.skip("langchain not installed (need llm tier)")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
def test_shared_module_imports():
    """shared 工具模块可正常导入."""
    from shared import get_api_key, gpu_summary, MockLLM
    summary = gpu_summary()
    assert isinstance(summary, str)
    mock = MockLLM()
    resp = mock.chat.completions.create(messages=[{"role": "user", "content": "test"}])
    assert resp.choices[0].message.content
    print(f"shared OK: gpu_summary={summary!r}")
