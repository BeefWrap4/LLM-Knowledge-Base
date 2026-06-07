# W3 LLM tier 实现计划 — Real API Code

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 把 199 个 `ch*/llm/*.py` 中所有 `is_mock / USE_REAL_API / provider == "mock"` 分支去掉；`UnifiedClient` 变默认；`MockLLM` 类从 `shared/` 挪到 `tests/_mocks/`；`ch18` 的 `run_real()/run_mock()` 双函数拆成两个文件。

**前置依赖：** W1 + W2 完成。

---

## 文件清单

### 重点修改（29+ 个）

- `code/ch18_llm_frameworks/llm/01-09_*.py` —— 9 个，拆双函数
- `code/ch18_llm_frameworks/llm/13-18_*.py` —— 6 个，去 `USE_REAL_API`
- `code/ch18_llm_frameworks/llm/35_memory_token_control.py` —— 1 个
- `code/ch22_data_eng/llm/04_self_instruct.py` —— 1 个
- `code/ch22_data_eng/llm/11_constitutional_ai.py` —— 1 个
- `code/ch29_context_engineering/llm/05_haystack_chat_pipeline.py` —— 1 个
- `code/ch29_context_engineering/llm/12_full_context_pipeline.py` —— 1 个
- `code/ch19_distributed/gpu/02_ddp_training.py` —— 1 个（仅审，stub）
- `code/ch19_distributed/gpu/03_fsdp_training.py` —— 1 个（仅审，stub）

### 审计（不改 ~170 个）

- 199 - 29 = 170 个 `ch*/llm/*.py` 用 grep 审计无 mock 残留

### 创建

- `code/tests/_mocks/demo_langchain_*.py`（4-5 个）—— 教学 demo 用

### 测试

- `code/tests/test_llm_tier_no_mock.py` —— 自动审计

---

## 任务 1：自动审计测试（先建测试，定位违规）

- [ ] **步骤 1：创建 `code/tests/test_llm_tier_no_mock.py`**

```python
# tests/test_llm_tier_no_mock.py
"""审计: llm tier 主流程不应有 mock 引用 (除 LLM_MOCK 单行开关外)."""
import re
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_PATTERN = re.compile(r'os\.environ\.get\("LLM_MOCK"\)')
FORBIDDEN_PATTERNS = [
    "is_mock", "USE_REAL_API", 'provider == "mock"',
    'provider=="mock"', "MockLLM()", "fake_llm", "FakeListChatModel",
    "from shared.mock_llm", "import mock_llm",
]


def test_no_mock_in_llm_tier():
    violations = []
    for chap_dir in (CODE_ROOT).glob("ch*/llm"):
        for py_file in chap_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            content = py_file.read_text(encoding="utf-8")
            for pat in FORBIDDEN_PATTERNS:
                if pat in content:
                    # 排除允许的单行开关
                    for line in content.split("\n"):
                        if pat in line and not ALLOWED_PATTERN.search(line):
                            violations.append(
                                f"{py_file.relative_to(CODE_ROOT)}:{content.split(chr(10)).index(line)+1}: 出现 '{pat}'"
                            )
                            break

    assert not violations, "llm tier 不应出现 mock 引用:\n" + "\n".join(violations[:20])
```

- [ ] **步骤 2：跑**

```bash
cd code
pytest tests/test_llm_tier_no_mock.py -v
```

预期：FAIL（列出所有违规文件，这就是任务清单）。

---

## 任务 2：拆分 ch18/01-09 双函数

每个文件按统一模板改：

```python
# 修改前
def run_real():
    from langchain_openai import ChatOpenAI
    ...

def run_mock():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    ...

if __name__ == "__main__":
    run_real()
    # run_mock()  # 默认注释
```

```python
# 修改后（主文件）
# 删除 run_real() 函数体内的 fake_llm 部分
def main():
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    prompt = ChatPromptTemplate.from_messages([...])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": "..."})
    print(f"Real API answer: {result}")

if __name__ == "__main__":
    main()
```

```python
# tests/_mocks/demo_langchain_basic_chain.py（新增）
"""LangChain Chain 离线 demo (用 FakeListChatModel, 仅教学)."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.fake_chat_models import FakeListChatModel

fake_llm = FakeListChatModel(responses=["GIL 是全局解释器锁..."])
prompt = ChatPromptTemplate.from_messages([...])
chain = prompt | fake_llm | StrOutputParser()
result = chain.invoke({"question": "什么是 GIL?"})
print(f"Mock answer: {result}")
```

- [ ] **步骤 1：建 `code/tests/_mocks/demo_langchain_basic_chain.py`（ch18/01 的 mock 对应）**

- [ ] **步骤 2：改 `code/ch18_llm_frameworks/llm/01_langchain_basic_chain.py`**

- [ ] **步骤 3：跑验证**

```bash
cd code
LLM_MOCK=1 pytest tests/_mocks/demo_langchain_basic_chain.py -v
python ch18_llm_frameworks/llm/01_langchain_basic_chain.py
```

- [ ] **步骤 4：循环处理 02-09**

- [ ] **步骤 5：跑 `pytest tests/test_llm_tier_no_mock.py -v` 验证违规数下降**

- [ ] **步骤 6：Commit**

```bash
git add code/ch18_llm_frameworks/llm/01-09_*.py code/tests/_mocks/demo_langchain_*.py
git commit -m "W3 ch18/01-09: split run_real/run_mock into main + tests/_mocks demo"
```

---

## 任务 3：去 ch18/13-18 + 35 的 `USE_REAL_API` 分支

每个文件按统一模板改：

```python
# 修改前
USE_REAL_API = os.environ.get("USE_REAL_API") == "1"
if USE_REAL_API:
    from shared.chatmodel_factory import make_chat_model
    ...
else:
    class _MockLLM:
        ...
    Settings.llm = _MockLLM()

# 修改后
from shared.chatmodel_factory import make_chat_model
real_llm = make_chat_model(framework="llama_index")
if real_llm is not None:
    Settings.llm = real_llm
else:
    raise RuntimeError("需要 LLM_PROVIDER + API Key. 见 README §硬件矩阵")
```

- [ ] **步骤 1：处理 `13_llamaindex_vectorstore_index.py`**

- [ ] **步骤 2：处理 14-18, 35**

- [ ] **步骤 3：跑 `pytest tests/test_llm_tier_no_mock.py -v`**

- [ ] **步骤 4：Commit**

```bash
git add code/ch18_llm_frameworks/llm/13-18_*.py code/ch18_llm_frameworks/llm/35_*.py
git commit -m "W3 ch18/13-18+35: remove USE_REAL_API branch, default to real LLM"
```

---

## 任务 4：去 ch22/04, 11 + ch29/05, 12 的 `if not is_real_api` 分支

模式同任务 3。

- [ ] **步骤 1：处理 4 个文件**

- [ ] **步骤 2：跑审计测试**

- [ ] **步骤 3：Commit**

---

## 任务 5：审 ch19/02-03 DDP/FSDP（仅审，stub）

W3 阶段不真实化（留 W6），仅审 + 去 mock import。

- [ ] **步骤 1：grep**

```bash
cd code
grep -n "mock\|fake" ch19_distributed/gpu/02_ddp_training.py ch19_distributed/gpu/03_fsdp_training.py
```

- [ ] **步骤 2：若仅有 stub 注释，不动；若真有 mock 类，删**

- [ ] **步骤 3：Commit**

---

## 任务 6：审剩余 ~170 个 llm/*.py

- [ ] **步骤 1：跑审计**

```bash
cd code
pytest tests/test_llm_tier_no_mock.py -v
```

预期：PASS 或违规数 = 0。

- [ ] **步骤 2：若还有违规，单独处理**

---

## 任务 7：手测 5 个代表性 LLM 文件

- [ ] **步骤 1：选 5 个**

候选：
- `ch13_prompt_engineering/llm/01_few_shot.py`
- `ch14_rag/llm/01_basic_rag.py`
- `ch15_agent/llm/01_react_basic.py`
- `ch17_evaluation/llm/01_llm_judge.py`
- `ch18_llm_frameworks/llm/01_langchain_basic_chain.py`

- [ ] **步骤 2：用真实 API 跑**

```bash
cd code
export DEEPSEEK_API_KEY=sk-real-test
for f in ch13_prompt_engineering/llm/01_few_shot.py \
         ch14_rag/llm/01_basic_rag.py \
         ch15_agent/llm/01_react_basic.py; do
  echo "=== $f ==="
  python "$f" 2>&1 | tail -3
done
```

预期：3 个文件真实调用 DeepSeek 并返回非空内容。

- [ ] **步骤 3：用 mock 跑同样 3 个**

```bash
LLM_MOCK=1 python ch13_prompt_engineering/llm/01_few_shot.py
LLM_MOCK=1 python ch14_rag/llm/01_basic_rag.py
LLM_MOCK=1 python ch15_agent/llm/01_react_basic.py
```

预期：3 个走 mock，无 Key 不抛错。

---

## 任务 8：跑 `make ci-llm` 全绿

- [ ] **步骤 1：跑**

```bash
cd code
make ci-llm
```

预期：~3 分钟内全绿（mock 路径）。

- [ ] **步骤 2：Commit 收尾**

```bash
git add -A
git commit -m "W3 llm tier: all 199 files audited, mock paths removed from main flow"
```

---

## W3 验收清单

- [ ] `tests/test_llm_tier_no_mock.py` 测试通过
- [ ] 9 个 ch18/01-09 文件拆为真实 + demo
- [ ] 6+1 个 ch18/13-18+35 文件去 `USE_REAL_API`
- [ ] 4 个 ch22/29 文件去 `if not is_real_api`
- [ ] 199 个 llm/*.py 全部审计无 mock 残留
- [ ] 5 个手测代表性文件真实 API + mock 双跑都通
- [ ] `make ci-llm` 全绿
