# ---
# code/tests/test_pilots.py
# Smoke tests for Wave 0-1 pilot + representative Ch02-11 examples
# ---
"""
每个例子必须能 `python file.py` 跑通且输出 "OK"。
跳过策略: 若 module 缺失 (ModuleNotFoundError), 自动 skip, 不算 fail.
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


def _has_module(module: str) -> bool:
    """检查某个 module 是否已安装 (用于 skip)."""
    return subprocess.run(
        [PY, "-c", f"import {module}"],
        capture_output=True,
    ).returncode == 0


# =============================================================================
# Wave 0 pilots
# =============================================================================

@pytest.mark.core
def test_ch01_list_dict_basics():
    """Ch01: 列表字典基础."""
    result = _run_example("ch01_python_basics/core/22_list_dict_basics.py")
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


# =============================================================================
# Wave 1 (Ch01-11 核心提取)  代表性 smoke tests
# =============================================================================

@pytest.mark.core
def test_ch01_python_313_features():
    """Ch01: Python 3.13 特性 (no-GIL 实验性)."""
    if sys.version_info < (3, 13):
        pytest.skip("Requires Python 3.13+")
    result = _run_example("ch01_python_basics/core/01_python_313_features.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"


@pytest.mark.core
def test_ch01_legb_rule():
    """Ch01: LEGB 作用域规则."""
    result = _run_example("ch01_python_basics/core/19_legb_rule.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
def test_ch02_object_identity():
    """Ch02: 对象身份/类型/值 区别."""
    result = _run_example("ch02_mutability/core/01_object_identity_type_value.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
def test_ch03_class_definition():
    """Ch03: 类的定义基础."""
    result = _run_example("ch03_oop/core/01_class_definition.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
def test_ch04_closure():
    """Ch04: 闭包基础."""
    result = _run_example("ch04_advanced_features/core/01_closure_basics.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
def test_ch05_gil_demo():
    """Ch05: GIL 切换间隔."""
    result = _run_example("ch05_concurrency/core/01_gil_switch_interval.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
def test_ch06_pymalloc():
    """Ch06: pymalloc 对象大小."""
    result = _run_example("ch06_memory_gc/core/01_pymalloc_object_size.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
def test_ch07_linked_list():
    """Ch07: 链表实现."""
    result = _run_example("ch07_data_structures/core/01_linked_list.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_ch08_ndarray_basics():
    """Ch08: NumPy ndarray 基础."""
    result = _run_example("ch08_data_science/core/01_ndarray_basics.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
@pytest.mark.skipif(not _has_module("fastapi"), reason="fastapi not installed")
def test_ch09_dependency_injection():
    """Ch09: FastAPI 依赖注入."""
    result = _run_example("ch09_fastapi/core/01_dependency_injection.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
@pytest.mark.skipif(not _has_module("sklearn"), reason="sklearn not installed")
def test_ch10_linear_regression():
    """Ch10: 线性回归."""
    result = _run_example("ch10_ml_basics/core/01_linear_regression.py", timeout=60)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
@pytest.mark.skipif(not _has_module("torch"), reason="torch not installed")
def test_ch11_tensor_ops():
    """Ch11: PyTorch tensor 基础运算."""
    result = _run_example("ch11_pytorch/core/01_tensor_ops.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


# =============================================================================
# Wave 2 (Ch12-24) — gpu/ 例子默认 skip, core/ 持续覆盖
# =============================================================================

@pytest.mark.core
@pytest.mark.skipif(not _has_module("torch"), reason="torch not installed")
def test_ch12_scaled_dot_product_attention():
    """Ch12: Scaled Dot-Product Attention."""
    result = _run_example("ch12_transformer_architecture/core/01_scaled_dot_product_attention.py")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.core
@pytest.mark.skipif(not _has_module("torch"), reason="torch not installed")
def test_ch12_multi_head_attention():
    """Ch12: Multi-Head Attention."""
    result = _run_example("ch12_transformer_architecture/core/02_multi_head_attention.py", timeout=60)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


@pytest.mark.llm
def test_ch18_langchain_chain_mock():
    """Ch18: LangChain basic chain demo (mock 模式, 不需 API key). W3 之后 mock 已下沉到 tests/_mocks/."""
    result = _run_example("tests/_mocks/demo_langchain_basic_chain.py", timeout=30)
    if "ModuleNotFoundError" in result.stderr and "langchain" in result.stderr:
        pytest.skip("langchain not installed (need llm tier)")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"stdout:\n{result.stdout}"


# =============================================================================
# shared utilities
# =============================================================================

@pytest.mark.core
def test_shared_module_imports():
    """shared 工具模块可正常导入.

    注: MockLLM 已迁移至 tests/_mocks/mock_llm.py (W1-T5).
    """
    from shared import get_api_key, gpu_summary
    from tests._mocks import MockLLM
    summary = gpu_summary()
    assert isinstance(summary, str)
    mock = MockLLM()
    resp = mock.chat.completions.create(messages=[{"role": "user", "content": "test"}])
    assert resp.choices[0].message.content
    print(f"shared OK: gpu_summary={summary!r}")


# =============================================================================
# GPU examples — 默认 skip, 仅在 GPU CI 启用
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(not _has_module("torch"), reason="torch not installed")
def test_ch12_pytorch_gpu_smoke():
    """Ch12: PyTorch GPU smoke (skip on Mac/laptop)."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    import torch
    x = torch.randn(2, 3, device="cuda")
    assert x.device.type == "cuda"
    assert x.shape == (2, 3)
    print(f"GPU OK: {x.device}")


# Late import for the gpu test above
try:
    import torch  # noqa: E402
except ImportError:
    torch = None

print("OK")
