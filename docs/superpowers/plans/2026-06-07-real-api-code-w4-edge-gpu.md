# W4 端侧 GPU 实现计划 — Real API Code

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 把 `ch28_edge_llm/gpu/` 10 个文件从 mock / 打印代码字符串改为真实调用（MLX / Ollama / llama.cpp / webllm / wasmtime）。

**前置依赖：** W1 + W2 + W3 完成。

**目标硬件：** Apple M-series（MLX / Ollama），通用 CPU（llama.cpp / wasmtime）

---

## 文件清单

### 修改

- `code/ch28_edge_llm/gpu/01_apple_mlx_basic.py`
- `code/ch28_edge_llm/gpu/02_mlx_unified_memory.py`
- `code/ch28_edge_llm/gpu/03_llama_cpp_gguf_quantization.py`
- `code/ch28_edge_llm/gpu/04_llama_cpp_backends.py`
- `code/ch28_edge_llm/gpu/05_ollama_http_api.py`
- `code/ch28_edge_llm/gpu/06_ollama_modelfile.py`
- `code/ch28_edge_llm/gpu/07_webllm_browser_inference.py`
- `code/ch28_edge_llm/gpu/08_webgpu_vs_wasm.py`
- `code/ch28_edge_llm/gpu/09_snapdragon_hexagon_npu.py`（保留教学性，不真实）
- `code/ch28_edge_llm/gpu/10_secure_minions_protocol.py`

### 测试

- `code/tests/test_ch28_hardware_checks.py`

---

## 任务 1：建硬件检查测试

- [ ] **步骤 1：创建 `code/tests/test_ch28_hardware_checks.py`**

```python
# tests/test_ch28_hardware_checks.py
"""验证 ch28 例子在缺硬件时抛错，在有硬件时不抛错。"""
import pytest
from unittest.mock import patch


def test_01_apple_mlx_requires_apple_silicon():
    """缺 Apple Silicon → 抛 RuntimeError."""
    from ch28_edge_llm.gpu import _01_apple_mlx_basic
    # 直接调用会 import mlx；改为测试函数 check_hw()
    assert hasattr(_01_apple_mlx_basic, "check_hardware")
    with patch("shared.gpu_guard._platform_system", return_value="Linux"):
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            _01_apple_mlx_basic.check_hardware()
```

（具体测试函数在每个 .py 改造时确定）

---

## 任务 2：改造 `01-02` MLX

- [ ] **步骤 1：读现状**

- [ ] **步骤 2：替换为真实 `mlx_lm.load + generate`**

```python
# 01_apple_mlx_basic.py 改造后
import sys
from pathlib import Path
_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_apple_silicon

def check_hardware():
    require_apple_silicon(min_memory_gb=8)

def main():
    check_hardware()
    from mlx_lm import load, generate
    model_path = "code/models/Qwen2.5-7B-Instruct-4bit-mlx"
    if not Path(model_path).exists():
        from shared._error_helper import raise_with_help
        raise_with_help(
            f"需要模型 {model_path}",
            "运行 `make download-models-edge`.",
        )
    model, tokenizer = load(model_path)
    prompt = "讲个笑话"
    response = generate(model, tokenizer, prompt=prompt, max_tokens=128)
    print(f"MLX response: {response}")

if __name__ == "__main__":
    main()
```

- [ ] **步骤 3：跑**

```bash
cd code
python ch28_edge_llm/gpu/01_apple_mlx_basic.py
```

预期：Apple M-series 上真实生成；其他平台抛 RuntimeError。

- [ ] **步骤 4：Commit**

---

## 任务 3：改造 `03-04` llama.cpp

模式同任务 2，区别是 `from llama_cpp import Llama`。

---

## 任务 4：改造 `05-06` Ollama（删除"打印代码字符串"）

- [ ] **步骤 1：删除 `show_native_api_call()` / `show_openai_compat_call()` 等"打印字符串"函数**

- [ ] **步骤 2：替换为真实 `httpx.post`**

- [ ] **步骤 3：`06_ollama_modelfile.py` 改为：生成 `code/models/Modelfile` + 真实 `subprocess.run(["ollama", "create", ...])`**

---

## 任务 5：改造 `07` webllm（用 playwright 真实打开）

```python
from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("https://mlc.ai/web-llm/")
        # 等待模型加载
        page.wait_for_selector("#chat-input", timeout=60000)
        page.fill("#chat-input", "Hello!")
        page.click("#send-button")
        # 截屏
        page.screenshot(path="code/.benchmarks/webllm-demo.png")
        browser.close()
```

---

## 任务 6：改造 `08` webgpu/wasm（用 wasmtime 跑真实 benchmark）

---

## 任务 7：`09` Snapdragon NPU 保留教学性

不改真实化（设备不可得），但加显式提示：

```python
def main():
    from shared._error_helper import raise_with_help
    raise_with_help(
        "此例子需要 Snapdragon NPU 设备.",
        "QNN SDK 仅在 Qualcomm 设备上可用. 详见 README §硬件矩阵.",
    )
```

---

## 任务 8：改造 `10` Secure Minions（TLS 协议模拟，Python 端可跑）

---

## 任务 9：跑 `make test-gpu` 验证

```bash
cd code
make test-gpu  # 仅 Apple M-series 上能跑
```

预期：10 个文件中能在 Apple Silicon 跑的都跑通，其他抛 RuntimeError 但不崩溃。

---

## 任务 10：Commit 收尾

```bash
git add -A
git commit -m "W4 ch28 edge GPU: real MLX/Ollama/llama.cpp/webllm calls"
```

---

## W4 验收清单

- [ ] 10 个 ch28/.../gpu/ 文件全部改造
- [ ] Apple M-series 笔记本：`make test-gpu` 跑通真实例子
- [ ] 其他硬件：抛 RuntimeError 明确提示
- [ ] 教程 28 章节中的代码块与文件一致
