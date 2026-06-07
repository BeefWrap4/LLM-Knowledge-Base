# W2 Core tier 实现计划 — Real API Code

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 审计 158 个 `ch*/core/*.py`，确保无隐藏的 mock / fake；删除任何 `from shared.mock_llm import ...` 残留；统一 `if __name__ == "__main__":` 段格式。

**前置依赖：** W1 基建完成。

---

## 文件清单

### 审计（不改）

158 个 `code/ch*/core/*.py`

### 可能创建

- `code/tests/test_core_no_mock.py` — 自动审计脚本（grep 验证）

---

## 任务 1：审计所有 core 文件，列出 mock / fake 残留

- [ ] **步骤 1：跑审计 grep**

```bash
cd code
grep -rln "MockLLM\|is_mock\|fake_llm\|mock_llm" ch*/core/*.py
grep -rln "import mock\|from shared.mock_llm" ch*/core/*.py
grep -rln "FakeListChatModel\|FakeChatModel" ch*/core/*.py
```

- [ ] **步骤 2：记录审计结果到 issue / commit message**

- [ ] **步骤 3：若发现任何文件含 mock 残留，单独处理**

---

## 任务 2：清理 `shared/__init__.py`（如未在 W1 完成）

- [ ] **步骤 1：grep 确认**

```bash
cd code
grep -n "mock_llm" shared/__init__.py
```

- [ ] **步骤 2：若存在，删除导入行**

---

## 任务 3：写自动审计测试

- [ ] **步骤 1：创建 `code/tests/test_core_no_mock.py`**

```python
# tests/test_core_no_mock.py
"""自动审计: core tier 任何文件不应有 mock 引用."""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_PATTERNS = ["MockLLM", "is_mock", "fake_llm",
                      "fake_chat", "FakeListChatModel"]


def test_no_mock_in_core_tier():
    violations = []
    for py_file in (CODE_ROOT / "ch01_python_basics" / "core").rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        content = py_file.read_text(encoding="utf-8")
        for pat in FORBIDDEN_PATTERNS:
            if pat in content:
                violations.append(f"{py_file.relative_to(CODE_ROOT)}: 出现 '{pat}'")

    assert not violations, "core tier 不应出现 mock 引用:\n" + "\n".join(violations)
```

- [ ] **步骤 2：跑测试**

```bash
cd code
pytest tests/test_core_no_mock.py -v
```

预期：PASS（无违规），或 FAIL（列出违规文件，需要单独修）。

---

## 任务 4：手测 10 个代表性 core 文件

- [ ] **步骤 1：选 10 个文件**

候选：
- `ch01_python_basics/core/01_hello.py`
- `ch05_concurrency/core/01_threading.py`
- `ch07_data_structures/core/01_list_dict.py`
- `ch08_data_science/core/01_numpy_basics.py`
- `ch08_data_science/core/03_pandas_basics.py`
- `ch10_ml_basics/core/01_linear_regression.py`
- `ch11_pytorch/core/01_tensor_basics.py`

- [ ] **步骤 2：逐个 `python xx.py` 跑通**

```bash
cd code
for f in ch01_python_basics/core/01_hello.py \
         ch05_concurrency/core/01_threading.py \
         ch08_data_science/core/01_numpy_basics.py; do
  echo "=== $f ==="
  python "$f" 2>&1 | tail -5
done
```

- [ ] **步骤 3：记录任何 import 错误或运行时错误到 issue**

---

## 任务 5：跑 `make ci-core` 全绿

- [ ] **步骤 1：跑**

```bash
cd code
make ci-core
```

预期：所有 core 文件在 60s 内跑通。

- [ ] **步骤 2：Commit 收尾**

```bash
git add -A
git commit -m "W2 core tier: audit passed, no mock residuals"
```

---

## W2 验收清单

- [ ] 158 个 core 文件全部审计，无 mock 残留
- [ ] `tests/test_core_no_mock.py` 测试通过
- [ ] 10 个手测代表性 core 文件 `python xx.py` 跑通
- [ ] `make ci-core` 全绿
