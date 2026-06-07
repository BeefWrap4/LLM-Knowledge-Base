# W1 基建实现计划 — Real API Code

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 改造 `shared/` 基建，让 `UnifiedClient` 缺 Key 抛错而非降级；扩展 `download_models.py` 覆盖 12+ 模型；建 `tests/_mocks/` 目录；让 `make ci-*` 默认走 mock；写 README 硬件矩阵。

**架构：** L3 共享层重写 + L4 资源准备 + L2 主流程前置条件齐备。

**技术栈：** Python 3.11+、openai SDK、ModelScope SDK、pytest。

**前置依赖：** 无（这是 wave 1）。  
**后续 wave 依赖：** W2-W8 全部依赖本 wave 完成。

---

## 文件清单

### 创建

- `code/shared/_error_helper.py` — 统一错误格式化（~40 行）
- `code/tests/_mocks/__init__.py` — 暴露 MockLLM 给 pytest
- `code/tests/_mocks/conftest.py` — pytest 自动设 LLM_MOCK=1
- `code/tests/_mocks/mock_embedding.py` — 抽离 ch18 的 _MockEmbed
- `code/tests/_mocks/mock_vllm.py` — 抽离 ch25/10 的 MockAsyncLLMEngine
- `code/tests/_mocks/mock_trt.py` — 抽离 ch25/08 的 Mock TRT build
- `code/tests/test_unified_client.py` — UnifiedClient 行为测试
- `code/tests/test_error_helper.py` — 错误格式测试
- `code/tests/test_gpu_guard.py` — 硬件检测函数测试
- `code/tests/test_download_models.py` — 模型清单测试
- `code/scripts/setup_env.sh` — `make llm-doctor --setup` 调用的 shell 脚本

### 修改

- `code/shared/llm_client.py` — 缺 Key 抛错
- `code/shared/provider_registry.py` — 缺 Key 抛错
- `code/shared/gpu_guard.py` — 加 3 个 require_* 函数
- `code/shared/__init__.py` — 移除 mock_llm 导出
- `code/scripts/llm_doctor.py` — 加 --setup 引导
- `code/scripts/download_models.py` — 扩到 12+ 模型
- `code/Makefile` — 加 LLM_MOCK=1 前缀
- `code/README.md` — 加硬件矩阵段

### 移动（git mv 保留历史）

- `code/shared/mock_llm.py` → `code/tests/_mocks/mock_llm.py`

---

## 任务 1：`shared/_error_helper.py` — 统一错误格式化

**文件：**
- 创建：`code/shared/_error_helper.py`
- 测试：`code/tests/test_error_helper.py`

- [ ] **步骤 1：编写失败的测试**

```python
# tests/test_error_helper.py
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd code && pytest tests/test_error_helper.py -v`
预期：FAIL，`ModuleNotFoundError: No module named 'shared._error_helper'`

- [ ] **步骤 3：编写最少实现代码**

```python
# shared/_error_helper.py
"""统一错误格式化 — 所有 RuntimeError 走这个出口."""
from typing import Optional


def format_error(
    message: str,
    hint: str,
    file_path: Optional[str] = None,
    line: Optional[int] = None,
) -> str:
    """生成统一格式的错误信息.

    格式:
        [ERROR] {file}:{line}  {message}
        [HELP]  {hint}
        [HELP]  或 `export LLM_MOCK=1` 用 mock 跑 (仅 CI/离线)
    """
    location = ""
    if file_path:
        location = f"{file_path}"
        if line is not None:
            location += f":{line}"
        location = f"{location}  "
    parts = [f"[ERROR] {location}{message}"]
    parts.append(f"[HELP]  {hint}")
    parts.append("[HELP]  或 `export LLM_MOCK=1` 用 mock 跑 (仅 CI/离线)")
    return "\n".join(parts)


def raise_with_help(message: str, hint: str, exc_class=RuntimeError) -> None:
    """抛带 help 信息的异常."""
    raise exc_class(format_error(message, hint))
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_error_helper.py -v`
预期：PASS（2 个测试通过）

- [ ] **步骤 5：Commit**

```bash
git add code/shared/_error_helper.py code/tests/test_error_helper.py
git commit -m "Add shared._error_helper for unified error formatting"
```

---

## 任务 2：`shared/gpu_guard.py` — 加 3 个 require_* 函数

**文件：**
- 修改：`code/shared/gpu_guard.py`
- 测试：`code/tests/test_gpu_guard.py`

- [ ] **步骤 1：阅读现有 `shared/gpu_guard.py`**

读取 `code/shared/gpu_guard.py`，了解现有 API（已知有 `gpu_guard()` 函数）。

- [ ] **步骤 2：编写失败的测试**

```python
# tests/test_gpu_guard.py
import pytest
from unittest.mock import patch, MagicMock
from shared.gpu_guard import require_nvidia_gpu, require_apple_silicon, require_ollama

def test_require_nvidia_gpu_no_cuda():
    with patch("shared.gpu_guard._has_torch", return_value=True), \
         patch("shared.gpu_guard._torch_cuda_available", return_value=False):
        with pytest.raises(RuntimeError, match="需要 NVIDIA GPU"):
            require_nvidia_gpu(min_vram_gb=8)

def test_require_nvidia_gpu_insufficient_vram():
    fake_props = MagicMock(total_memory=4 * 1e9)  # 4GB
    with patch("shared.gpu_guard._has_torch", return_value=True), \
         patch("shared.gpu_guard._torch_cuda_available", return_value=True), \
         patch("shared.gpu_guard._torch_device_count", return_value=1), \
         patch("shared.gpu_guard._torch_device_props", return_value=fake_props):
        with pytest.raises(RuntimeError, match="显存.*4.0GB.*< 8GB"):
            require_nvidia_gpu(min_vram_gb=8)

def test_require_nvidia_gpu_sufficient():
    fake_props = MagicMock(total_memory=24 * 1e9)
    with patch("shared.gpu_guard._has_torch", return_value=True), \
         patch("shared.gpu_guard._torch_cuda_available", return_value=True), \
         patch("shared.gpu_guard._torch_device_count", return_value=1), \
         patch("shared.gpu_guard._torch_device_props", return_value=fake_props):
        require_nvidia_gpu(min_vram_gb=8)  # 不抛错

def test_require_apple_silicon_on_linux():
    with patch("shared.gpu_guard._platform_system", return_value="Linux"):
        with pytest.raises(RuntimeError, match="需要 Apple Silicon"):
            require_apple_silicon()

def test_require_ollama_not_running():
    with patch("shared.gpu_guard._httpx_get", side_effect=Exception("ConnectError")):
        with pytest.raises(RuntimeError, match="Ollama 未运行"):
            require_ollama("llama3.2:3b")
```

- [ ] **步骤 3：运行测试验证失败**

运行：`pytest tests/test_gpu_guard.py -v`
预期：FAIL，`ImportError: cannot import name 'require_nvidia_gpu'`

- [ ] **步骤 4：实现 3 个函数**

在 `code/shared/gpu_guard.py` **追加**（不要修改现有 `gpu_guard()` 函数）：

```python
# 在 gpu_guard.py 末尾追加:

def _has_torch() -> bool:
    try:
        import torch  # noqa
        return True
    except ImportError:
        return False


def _torch_cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


def _torch_device_count() -> int:
    import torch
    return torch.cuda.device_count()


def _torch_device_props(idx: int):
    import torch
    return torch.cuda.get_device_properties(idx)


def _platform_system() -> str:
    import platform
    return platform.system()


def _platform_machine() -> str:
    import platform
    return platform.machine()


def _httpx_get(url: str, timeout: float = 2.0):
    import httpx
    return httpx.get(url, timeout=timeout)


def require_nvidia_gpu(min_vram_gb: int = 8, min_count: int = 1) -> None:
    """检查 NVIDIA GPU + 显存 + 数量. 不足抛 RuntimeError."""
    from shared._error_helper import raise_with_help
    if not _has_torch():
        raise_with_help(
            "此例子需要 torch. 运行 `make install-gpu`.",
            "无 torch 时无法检测 GPU.",
        )
    if not _torch_cuda_available():
        raise_with_help(
            f"此例子需要 NVIDIA GPU (≥{min_count} 张, ≥{min_vram_gb}GB). 当前未检测到 CUDA.",
            "详见 README §硬件 × 章节矩阵.",
        )
    count = _torch_device_count()
    if count < min_count:
        raise_with_help(
            f"需要 ≥{min_count} 张 GPU, 当前 {count} 张.",
            "详见 README §硬件 × 章节矩阵.",
        )
    if min_vram_gb:
        for i in range(count):
            total = _torch_device_props(i).total_memory / 1e9
            if total < min_vram_gb:
                raise_with_help(
                    f"GPU {i} 显存 {total:.1f}GB < {min_vram_gb}GB.",
                    "需要更大显存. 详见 README §硬件矩阵.",
                )


def require_apple_silicon(min_memory_gb: int = 8) -> None:
    """检查 Apple Silicon. 否则抛 RuntimeError."""
    from shared._error_helper import raise_with_help
    if _platform_system() != "Darwin" or _platform_machine() != "arm64":
        raise_with_help(
            "此例子需要 Apple Silicon (M-series Mac).",
            "详见 README §硬件 × 章节矩阵.",
        )


def require_ollama(model: str = "llama3.2:3b") -> None:
    """检查 Ollama 服务 + 模型. 否则抛 RuntimeError."""
    from shared._error_helper import raise_with_help
    try:
        r = _httpx_get("http://localhost:11434/api/tags", timeout=2.0)
        r.raise_for_status()
        models = [m["name"] for m in r.json()["models"]]
        if not any(model in m for m in models):
            raise_with_help(
                f"Ollama 已运行, 但缺模型 {model}.",
                f"运行 `ollama pull {model}`.",
            )
    except Exception as e:
        raise_with_help(
            "Ollama 未运行. 先 `ollama serve`.",
            "或使用云端 LLM 替代. 详见 README §硬件 × 章节矩阵.",
        )
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_gpu_guard.py -v`
预期：PASS（5 个测试通过）

- [ ] **步骤 6：Commit**

```bash
git add code/shared/gpu_guard.py code/tests/test_gpu_guard.py
git commit -m "Add shared.gpu_guard require_* functions (nvidia/apple/ollama)"
```

---

## 任务 3：`shared/llm_client.py` — 缺 Key 抛错

**文件：**
- 修改：`code/shared/llm_client.py:55-76`（`__init__` 缺 Key 降级分支）
- 测试：`code/tests/test_unified_client.py`

- [ ] **步骤 1：编写失败的测试**

```python
# tests/test_unified_client.py
import os
import pytest
from shared.llm_client import UnifiedClient


def test_unified_client_no_key_raises(monkeypatch):
    """缺 API Key 必须抛 RuntimeError，不再降级 mock."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(RuntimeError, match="缺 API Key"):
        UnifiedClient()


def test_unified_client_dummy_key_raises(monkeypatch):
    """占位 key 'YOUR_API_KEY' 也要抛错."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "YOUR_API_KEY")
    with pytest.raises(RuntimeError, match="缺 API Key"):
        UnifiedClient()


def test_unified_client_with_key_succeeds(monkeypatch):
    """有真实 Key 时不抛错."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-real-test-key")
    client = UnifiedClient(provider="deepseek")
    assert client.api_key == "sk-real-test-key"
    assert client.is_mock is False


def test_unified_client_mock_env_var(monkeypatch):
    """LLM_MOCK=1 走 mock，不抛错."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MOCK", "1")
    # 暂时这个测试会失败，W1 任务 5 之后通过
    client = UnifiedClient(provider="deepseek")
    assert client.is_mock is True
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_unified_client.py -v`
预期：FAIL（`test_unified_client_no_key_raises` 失败，因为当前实现是降级 mock 不抛错）

- [ ] **步骤 3：修改 `UnifiedClient.__init__` 缺 Key 行为**

修改 `code/shared/llm_client.py:55-76`：

```python
# 修改前 (当前)
        if not self.api_key or self.api_key == "YOUR_API_KEY":
            print(
                f"[WARN] UnifiedClient: 无 {self.provider.env_key}, 降级到 MockLLM (provider={self.provider.name})",
                file=sys.stderr,
            )
            self.client = None
        elif self.provider.api_style == "openai" and HAS_OPENAI:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider.base_url,
                timeout=timeout,
            )
        else:
            print(
                f"[WARN] UnifiedClient: provider={self.provider.name} 不支持或 openai SDK 缺失, 降级到 mock",
                file=sys.stderr,
            )
            self.client = None

# 修改后
        # LLM_MOCK=1 环境变量 → 强制走 mock (CI/离线)
        if os.environ.get("LLM_MOCK") == "1":
            self.client = None
            return
        if not self.api_key or self.api_key == "YOUR_API_KEY":
            from shared._error_helper import raise_with_help
            raise_with_help(
                f"厂商 {self.provider.name} 缺 API Key (env {self.provider.env_key}).",
                "运行 `make llm-doctor` 诊断; 或参考 README §环境配置.",
            )
        if self.provider.api_style == "openai" and HAS_OPENAI:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider.base_url,
                timeout=timeout,
            )
        elif self.provider.api_style == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key, timeout=timeout)
            except ImportError:
                from shared._error_helper import raise_with_help
                raise_with_help(
                    f"厂商 {self.provider.name} 需 anthropic SDK.",
                    "运行 `pip install anthropic`.",
                )
        else:
            from shared._error_helper import raise_with_help
            raise_with_help(
                f"厂商 {self.provider.name} 不支持或 openai SDK 缺失.",
                "运行 `make install-llm`.",
            )
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_unified_client.py -v`
预期：PASS（4 个测试通过）

- [ ] **步骤 5：Commit**

```bash
git add code/shared/llm_client.py code/tests/test_unified_client.py
git commit -m "UnifiedClient: raise on missing key (no more silent mock fallback)"
```

---

## 任务 4：`shared/provider_registry.py` — 缺 Key 抛错

**文件：**
- 修改：`code/shared/provider_registry.py:142-162`（`get_default_provider`）
- 测试：扩展 `code/tests/test_unified_client.py`

- [ ] **步骤 1：编写失败的测试**

```python
# tests/test_unified_client.py 追加
def test_get_default_provider_no_key_raises(monkeypatch):
    from shared.provider_registry import get_default_provider
    for k in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "KIMI_API_KEY",
              "SILICONFLOW_API_KEY", "MINIMAX_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(RuntimeError, match="缺 API Key"):
        get_default_provider()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_unified_client.py::test_get_default_provider_no_key_raises -v`
预期：FAIL

- [ ] **步骤 3：修改 `get_default_provider`**

修改 `code/shared/provider_registry.py:142-162`：

```python
# 修改前
def get_default_provider() -> Provider:
    """从环境变量推断默认厂商.

    优先级:
      1. LLM_PROVIDER 环境变量
      2. 第一个有 Key 的国内厂商 (deepseek → kimi → siliconflow)
      3. 第一个有 Key 的海外厂商 (openai → anthropic)
      4. mock (最后兜底)
    """
    env_choice = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if env_choice and env_choice in PROVIDERS:
        return PROVIDERS[env_choice]

    for p in list_providers():
        if p.region == "CN" and p.has_key():
            return p
    for p in list_providers():
        if p.region == "US" and p.has_key():
            return p
    return PROVIDERS["mock"]


# 修改后
def get_default_provider() -> Provider:
    """从环境变量推断默认厂商.

    优先级:
      1. LLM_PROVIDER 环境变量
      2. 第一个有 Key 的国内厂商 (deepseek → kimi → siliconflow)
      3. 第一个有 Key 的海外厂商 (openai → anthropic)
      4. 抛 RuntimeError (不再降级 mock)

    注: mock 路径通过 LLM_MOCK=1 触发，不由本函数负责。
    """
    from shared._error_helper import raise_with_help
    env_choice = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if env_choice and env_choice in PROVIDERS:
        p = PROVIDERS[env_choice]
        if p.name == "mock":
            return p  # 用户显式选 mock
        if not p.has_key():
            raise_with_help(
                f"LLM_PROVIDER={env_choice} 但缺 API Key (env {p.env_key}).",
                "运行 `make llm-doctor` 诊断; 或 `export LLM_MOCK=1`.",
            )
        return p

    for p in list_providers():
        if p.region == "CN" and p.has_key():
            return p
    for p in list_providers():
        if p.region == "US" and p.has_key():
            return p

    raise_with_help(
        "未配置任何 LLM 厂商 API Key.",
        "运行 `make llm-doctor` 诊断; 或 `export LLM_MOCK=1` 用 mock (仅 CI).",
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_unified_client.py -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add code/shared/provider_registry.py code/tests/test_unified_client.py
git commit -m "provider_registry: raise on no key (mock only via LLM_MOCK=1)"
```

---

## 任务 5：移动 `shared/mock_llm.py` 到 `tests/_mocks/`

**文件：**
- 移动：`code/shared/mock_llm.py` → `code/tests/_mocks/mock_llm.py`
- 创建：`code/tests/_mocks/__init__.py`
- 创建：`code/tests/_mocks/conftest.py`
- 修改：`code/shared/__init__.py` 移除 `from .mock_llm import ...`

- [ ] **步骤 1：使用 git mv 移动文件**

```bash
cd code
git mv shared/mock_llm.py tests/_mocks/mock_llm.py
```

- [ ] **步骤 2：创建 `tests/_mocks/__init__.py`**

```python
# tests/_mocks/__init__.py
"""Mock implementations for CI/离线测试.

仅 pytest 自动加载 (conftest.py 设 LLM_MOCK=1) 时使用.
主流程不导入此模块.
"""
from tests._mocks.mock_llm import MockLLM, deterministic_response

__all__ = ["MockLLM", "deterministic_response"]
```

- [ ] **步骤 3：创建 `tests/_mocks/conftest.py`**

```python
# tests/_mocks/conftest.py
"""Pytest 自动配置: 设 LLM_MOCK=1 让所有测试走 mock."""
import os
os.environ.setdefault("LLM_MOCK", "1")
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-test")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-dummy-test")
```

- [ ] **步骤 4：修改 `code/shared/__init__.py`**

查找 `code/shared/__init__.py` 当前内容，移除 `from .mock_llm import ...` 行（如果有）。

- [ ] **步骤 5：运行所有测试验证不破**

```bash
cd code
pytest tests/ -m "not gpu" -v
```

预期：所有现有测试仍 PASS（mock 路径工作正常）。

- [ ] **步骤 6：Commit**

```bash
git add code/shared/mock_llm.py code/tests/_mocks/ code/shared/__init__.py
git commit -m "Move shared/mock_llm.py to tests/_mocks/ (CI-only mock path)"
```

---

## 任务 6：扩展 `scripts/download_models.py` 覆盖 12+ 模型

**文件：**
- 修改：`code/scripts/download_models.py:14-22`（`MODELS_TO_DOWNLOAD` 字典）
- 测试：`code/tests/test_download_models.py`

- [ ] **步骤 1：编写失败的测试**

```python
# tests/test_download_models.py
from scripts.download_models import MODELS_TO_DOWNLOAD


def test_models_dict_has_minimum_12():
    """教程需要至少 12 个模型."""
    assert len(MODELS_TO_DOWNLOAD) >= 12


def test_models_dict_keys():
    """每个模型必须有 model_id + local_name + size_gb + tier."""
    for key, info in MODELS_TO_DOWNLOAD.items():
        assert "model_id" in info, f"{key} 缺 model_id"
        assert "local_name" in info, f"{key} 缺 local_name"
        assert "size_gb" in info, f"{key} 缺 size_gb"
        assert "tier" in info, f"{key} 缺 tier"


def test_models_dict_contains_known():
    """已知必须有的模型."""
    required = [
        "bge-small-zh", "bge-reranker", "qwen0_5b",
        "qwen7b", "llama8b", "cosmos7b",
        "pi0-vla", "r1-distill-1_5b", "mlx-qwen7b-4bit",
        "llama-cpp-3b",
    ]
    for r in required:
        assert r in MODELS_TO_DOWNLOAD, f"缺 {r}"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_download_models.py -v`
预期：FAIL

- [ ] **步骤 3：扩展模型字典**

修改 `code/scripts/download_models.py:14-22`，把简单字典替换为：

```python
MODELS_TO_DOWNLOAD = {
    # === Embedding / Rerank (默认下载) ===
    "bge-small-zh": {
        "model_id": "BAAI/bge-small-zh-v1.5",
        "local_name": "bge-small-zh-v1.5",
        "size_gb": 0.1,
        "tier": "embedding",
        "chapters": ["ch14_rag", "ch17_evaluation", "ch20_llmops", "ch22_data_eng"],
        "required": True,
    },
    "bge-reranker": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "local_name": "bge-reranker-v2-m3",
        "size_gb": 0.6,
        "tier": "reranker",
        "chapters": ["ch17_evaluation", "ch22_data_eng"],
        "required": True,
    },
    # === LLM 小模型 (默认下载) ===
    "qwen0_5b": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_name": "Qwen2.5-0.5B-Instruct",
        "size_gb": 1.0,
        "tier": "llm-small",
        "chapters": ["ch12_transformer_architecture", "ch13_prompt_engineering",
                     "ch14_rag", "ch15_agent", "ch16_finetuning",
                     "ch17_evaluation", "ch18_llm_frameworks",
                     "ch19_distributed", "ch29_context_engineering"],
        "required": True,
    },
    # === LLM 中等 (--llm 时下载) ===
    "qwen7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_name": "Qwen2.5-7B-Instruct",
        "size_gb": 15.0,
        "tier": "llm-medium",
        "chapters": ["ch25_inference_engines"],
        "required": False,
    },
    "llama8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_name": "Llama-3.1-8B-Instruct",
        "size_gb": 16.0,
        "tier": "llm-medium",
        "chapters": ["ch25_inference_engines"],
        "required": False,
        "needs_auth": True,
    },
    "cosmos7b": {
        "model_id": "nvidia/Cosmos-1.0-7B",
        "local_name": "Cosmos-1.0-7B",
        "size_gb": 14.0,
        "tier": "world-model",
        "chapters": ["ch26_world_models"],
        "required": False,
    },
    "pi0-vla": {
        "model_id": "lerobot/pi0-base",
        "local_name": "Pi0-VLA-base",
        "size_gb": 8.0,
        "tier": "vla",
        "chapters": ["ch26_world_models"],
        "required": False,
    },
    "r1-distill-1_5b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "local_name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "size_gb": 3.0,
        "tier": "reasoner",
        "chapters": ["ch27_reasoning_ttc"],
        "required": False,
    },
    "mlx-qwen7b-4bit": {
        "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "local_name": "Qwen2.5-7B-Instruct-4bit-mlx",
        "size_gb": 5.0,
        "tier": "edge-mlx",
        "chapters": ["ch28_edge_llm"],
        "required": False,
        "platform": "apple-silicon",
    },
    "llama-cpp-3b": {
        "model_id": "TheBloke/Llama-3.2-3B-Instruct-GGUF",
        "local_name": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.0,
        "tier": "edge-gguf",
        "chapters": ["ch28_edge_llm"],
        "required": False,
    },
    # === 训练辅助 (复用 qwen0_5b) ===
    "qwen0_5b-lora": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_name": "Qwen2.5-0.5B-Instruct-lora",
        "size_gb": 0.1,  # 仅 LoRA adapters
        "tier": "training",
        "chapters": ["ch16_finetuning"],
        "required": False,
        "depends_on": "qwen0_5b",
    },
    "qwen0_5b-ddp": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_name": "Qwen2.5-0.5B-Instruct-ddp",
        "size_gb": 1.0,
        "tier": "training",
        "chapters": ["ch19_distributed"],
        "required": False,
        "depends_on": "qwen0_5b",
    },
}
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_download_models.py -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add code/scripts/download_models.py code/tests/test_download_models.py
git commit -m "Expand download_models.py to 12+ models with chapter mapping"
```

---

## 任务 7：`scripts/llm_doctor.py` — 加 `--setup` 引导

**文件：**
- 修改：`code/scripts/llm_doctor.py`
- 创建：`code/scripts/setup_env.sh`

- [ ] **步骤 1：阅读 `code/scripts/llm_doctor.py` 现状**

- [ ] **步骤 2：在 `llm_doctor.py` 追加 `--setup` argparse**

```python
# 在 main() 函数入口追加:
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true",
                        help="交互式引导配置 API Key")
    parser.add_argument("--check", action="store_true",
                        help="测试已配置 Key 是否有效")
    args = parser.parse_args()

    if args.setup:
        setup_wizard()
    elif args.check:
        check_keys()
    else:
        # 现有 report 行为
        ...
```

- [ ] **步骤 3：实现 `setup_wizard()` 函数**

在 `code/scripts/llm_doctor.py` 追加：

```python
def setup_wizard():
    """交互式引导: 帮用户配置 API Key."""
    print("=" * 60)
    print("LLM API Key 配置向导")
    print("=" * 60)
    print()
    print("教程默认使用 DeepSeek (国内访问快 + 注册送 ¥10 + OpenAI 协议).")
    print("注册地址: https://platform.deepseek.com")
    print()
    print("其他可选厂商:")
    for p in list_providers():
        if p.name in ("deepseek", "mock"):
            continue
        print(f"  - {p.display_name} ({p.name}): {p.free_tier}")
    print()

    choice = input("选择厂商 [1=DeepSeek, 2=其他, q=退出]: ").strip()
    if choice == "q":
        return

    if choice == "1":
        provider = "deepseek"
        env_var = "DEEPSEEK_API_KEY"
        print(f"\n请访问 https://platform.deepseek.com 注册并获取 API Key.")
    elif choice == "2":
        names = [p.name for p in list_providers() if p.name not in ("mock",)]
        for i, n in enumerate(names, 1):
            print(f"  {i}. {n}")
        idx = int(input("选编号: ")) - 1
        provider = names[idx]
        env_var = get_provider(provider).env_key
    else:
        print("无效选择.")
        return

    api_key = input(f"粘贴 {env_var} (输入时不会显示): ").strip()

    # 写入 .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    line = f"{env_var}={api_key}\n"
    if env_path.exists():
        content = env_path.read_text()
        if env_var in content:
            content = re.sub(rf"^{env_var}=.*$", line.rstrip(), content, flags=re.M)
        else:
            content += "\n" + line
        env_path.write_text(content)
    else:
        env_path.write_text(line)

    print(f"\n✅ 已写入 {env_path}")
    print(f"   {env_var}={'*' * 8}{api_key[-4:]}")

    # 测试
    print(f"\n测试调用...")
    test_result = test_provider(provider, api_key)
    if test_result["ok"]:
        print(f"✅ {provider} 可用 (延迟 {test_result['latency_ms']}ms)")
    else:
        print(f"❌ {provider} 失败: {test_result['error']}")


def test_provider(provider: str, api_key: str) -> dict:
    """测试厂商 API 是否可用."""
    import time
    from shared.llm_client import UnifiedClient
    client = UnifiedClient(provider=provider, api_key=api_key)
    t0 = time.time()
    try:
        resp = client.chat(prompt="回复 OK", max_tokens=10)
        latency = (time.time() - t0) * 1000
        return {"ok": True, "latency_ms": round(latency)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def check_keys():
    """列出已配置 Key 并测试连通性."""
    from shared.provider_registry import available_providers
    print("已配置 API Key 的厂商:")
    for p in available_providers():
        print(f"  ✓ {p.name:12s} ({p.display_name})")
        result = test_provider(p.name, os.environ.get(p.env_key))
        status = "✅" if result["ok"] else "❌"
        print(f"     {status} {result.get('latency_ms', result.get('error', ''))}ms" if result["ok"] else f"     {status} {result['error']}")
```

- [ ] **步骤 4：手测 `--setup` 流程（用 dummy key）**

```bash
cd code
echo "" | python scripts/llm_doctor.py --setup
# 应进入引导，但不写入 .env（用户回车跳过）
```

预期：脚本能进入 `setup_wizard()`，不崩溃。

- [ ] **步骤 5：Commit**

```bash
git add code/scripts/llm_doctor.py
git commit -m "Add llm_doctor --setup wizard and --check command"
```

---

## 任务 8：`Makefile` — `test-llm / ci-llm` 默认设 `LLM_MOCK=1`

**文件：**
- 修改：`code/Makefile`

- [ ] **步骤 1：修改 `test-llm` 和 `ci-llm` 目标**

```makefile
# 修改前
test-llm:  ## 跑 core + llm tier tests (用 mock_llm)
	pytest tests/ -m "not gpu" -q --tb=short

ci-llm:  ## 仅 llm tier — ~3 min
	python scripts/run_all_examples.py --tier llm --parallel 4 --timeout 180

# 修改后
test-llm:  ## 跑 core + llm tier tests (mock 模式, 无需 API Key)
	LLM_MOCK=1 pytest tests/ -m "not gpu" -q --tb=short

ci-llm:  ## 仅 llm tier — ~3 min (mock 模式, 无需 API Key)
	LLM_MOCK=1 python scripts/run_all_examples.py --tier llm --parallel 4 --timeout 180
```

- [ ] **步骤 2：新增目标 `llm-doctor-setup` 和 `ci-real`**

```makefile
# 在 Makefile 末尾追加:

llm-doctor-setup:  ## 交互式配置 API Key
	python scripts/llm_doctor.py --setup

llm-doctor-check:  ## 测试已配置 Key
	python scripts/llm_doctor.py --check

ci-real:  ## 夜间跑真实 API (需 DEEPSEEK_API_KEY)
	python scripts/run_all_examples.py --tier llm-real --parallel 1 --timeout 300

download-models-list:  ## 列出可下载的模型
	python scripts/download_models.py --list

download-models-default:  ## 下载默认必装模型 (embedding + reranker + qwen0.5b, ~1.7GB)
	python scripts/download_models.py --default

download-models-llm:  ## 下载 7B/8B LLM (vLLM 用, ~31GB)
	python scripts/download_models.py --tier llm-medium

download-models-gpu:  ## 下载 GPU 例子用模型 (cosmos / pi0 / r1-distill, ~25GB)
	python scripts/download_models.py --tier world-model --tier vla --tier reasoner

download-models-edge:  ## 下载端侧模型 (MLX / GGUF, ~7GB)
	python scripts/download_models.py --tier edge
```

- [ ] **步骤 3：手测 `make test-llm`**

```bash
cd code
make test-llm
```

预期：所有测试通过，输出 `LLM_MOCK=1` 在环境变量中。

- [ ] **步骤 4：手测 `make llm-doctor-setup --help`**

```bash
make llm-doctor-setup
# 看到引导 prompt
```

- [ ] **步骤 5：Commit**

```bash
git add code/Makefile
git commit -m "Makefile: LLM_MOCK=1 default for test-llm/ci-llm, add download-* targets"
```

---

## 任务 9：`code/README.md` — 加硬件 × 章节矩阵

**文件：**
- 修改：`code/README.md`

- [ ] **步骤 1：阅读 `code/README.md` 现状**

- [ ] **步骤 2：在 `code/README.md` 末尾追加"硬件 × 章节矩阵"段**

```markdown
## 🖥️ 硬件 × 章节矩阵

按你的硬件选章节（其他章节也能跑，但需要 API Key）：

| 硬件 | 可跑章节 | 必装命令 |
|------|---------|---------|
| **任意笔记本** (无 GPU) | Ch1-24 全部 core + llm tier | `pip install -r requirements-llm.txt` |
| **+ DeepSeek API Key** | 同上 + 真实 LLM 输出 | `make llm-doctor-setup` |
| **Apple M-series** (≥8GB) | Ch28 端侧 (MLX / Ollama) | `brew install ollama && ollama pull llama3.2:3b` |
| **NVIDIA GPU 8GB** | + Ch19 (DDP 0.5B) + Ch16 (LoRA 0.5B) | `make install-gpu` |
| **NVIDIA GPU 24GB+** | + Ch25 (vLLM 7B) + Ch27 (R1-Distill 1.5B) | `make download-models-llm` |
| **NVIDIA GPU 80GB (A100/H100)** | + Ch26 (Cosmos 7B + Pi0) | `make download-models-gpu` |

### 下载脚本速查

```bash
make download-models-list          # 列出所有可选模型
make download-models-default       # 1.7GB: embedding + reranker + 0.5B LLM
make download-models-llm           # +30GB: 7B/8B LLM (vLLM 用)
make download-models-gpu           # +25GB: 世界模型 / VLA / 推理
make download-models-edge          # +7GB: MLX / GGUF 端侧
```

### 各章节最低硬件需求

| 章节 | 最低硬件 | 模型权重 | 推荐 API |
|------|---------|---------|---------|
| Ch1-11 | 任意 | 无 | 无 |
| Ch12 Transformer | 任意 | 无 | 无 |
| Ch13 Prompt | 任意 | 无 | DeepSeek / OpenAI |
| Ch14 RAG | 任意 | bge-small-zh | DeepSeek |
| Ch15 Agent | 任意 | 无 | DeepSeek / Anthropic |
| Ch16 微调 | NVIDIA 8GB | qwen0.5b | DeepSeek |
| Ch17 评估 | 任意 | bge-small-zh + reranker | DeepSeek |
| Ch18 LLM 框架 | 任意 | 无 | DeepSeek / OpenAI |
| Ch19 分布式 | NVIDIA 24GB × 2 | qwen0.5b × 2 | DeepSeek |
| Ch20 LLMOps | 任意 | 无 | DeepSeek |
| Ch21-23 | 任意 | bge (部分) | DeepSeek |
| Ch24 云原生 | 任意 | 无 | DeepSeek |
| Ch25 推理引擎 | NVIDIA 24GB | qwen7b | 无 (本地) |
| Ch26 世界模型 | NVIDIA 80GB | cosmos7b + pi0 | 无 (本地) |
| Ch27 推理模型 | NVIDIA 24GB | r1-distill-1.5b | DeepSeek-R1 API |
| Ch28 端侧 | Apple Silicon | mlx-qwen7b / gguf-llama3b | Ollama (本地) |
| Ch29 Context Eng | 任意 | 无 | DeepSeek |

### 完整环境配置

```bash
# 1. 安装依赖
make install-llm        # 5 分钟, 无需 GPU

# 2. 配置 API Key (推荐 DeepSeek)
make llm-doctor-setup   # 交互式

# 3. 下载模型权重
make download-models-default  # 1.7GB, 必须

# 4. 验证
LLM_MOCK=1 make test-llm     # mock 测试
DEEPSEEK_API_KEY=sk-xxx python ch15_agent/llm/01_react_basic.py  # 真实跑
```
```

- [ ] **步骤 3：Commit**

```bash
git add code/README.md
git commit -m "README: add hardware × chapter matrix and download script cheatsheet"
```

---

## 任务 10：端到端验证 W1

- [ ] **步骤 1：跑 `make llm-doctor --check`（无 Key）**

```bash
cd code
make llm-doctor-check 2>&1 | head -20
```

预期：报错"未配置 API Key"，明确指向 `make llm-doctor-setup`。

- [ ] **步骤 2：跑 `make test-llm`（mock 模式）**

```bash
make test-llm
```

预期：所有测试 PASS（mock 路径工作正常）。

- [ ] **步骤 3：跑无 Key 真实 API 例子，预期抛错**

```bash
unset DEEPSEEK_API_KEY
unset OPENAI_API_KEY
unset LLM_MOCK
cd code
python -c "from shared.llm_client import UnifiedClient; UnifiedClient()"
```

预期：抛 `RuntimeError`，信息包含 "缺 API Key" 和 "make llm-doctor"。

- [ ] **步骤 4：跑有 Key 真实 API 例子**

```bash
export DEEPSEEK_API_KEY=sk-real-test  # 用真实 key 替换
python -c "from shared.llm_client import UnifiedClient; c = UnifiedClient(); print(c.model, c.is_mock)"
```

预期：打印 `deepseek-chat False`（不抛错）。

- [ ] **步骤 5：跑 LLM_MOCK=1 例子**

```bash
export LLM_MOCK=1
python -c "from shared.llm_client import UnifiedClient; c = UnifiedClient(); print(c.is_mock)"
```

预期：打印 `True`（走 mock）。

- [ ] **步骤 6：跑 `make download-models-list`**

```bash
cd code
make download-models-list
```

预期：列出 12+ 模型，分章节标注 required / optional。

- [ ] **步骤 7：全量测试 + 提交最终 commit**

```bash
make test-llm
git add -A
git commit -m "W1 infra: end-to-end verification passed" --allow-empty
```

---

## W1 验收清单

- [ ] `code/shared/_error_helper.py` 存在并被引用
- [ ] `code/shared/llm_client.py` 缺 Key 抛错（task 3）
- [ ] `code/shared/provider_registry.py` 缺 Key 抛错（task 4）
- [ ] `code/shared/gpu_guard.py` 有 3 个 `require_*` 函数（task 2）
- [ ] `code/shared/mock_llm.py` 已移到 `code/tests/_mocks/`（task 5）
- [ ] `code/scripts/llm_doctor.py` 有 `--setup` 和 `--check`（task 7）
- [ ] `code/scripts/download_models.py` 有 12+ 模型（task 6）
- [ ] `code/Makefile` `test-llm` / `ci-llm` 默认 `LLM_MOCK=1`（task 8）
- [ ] `code/README.md` 有"硬件 × 章节矩阵"段（task 9）
- [ ] 所有 pytest 测试通过
- [ ] 端到端 4 类场景验证通过（task 10）
