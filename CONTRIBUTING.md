# Contributing to Python 到大模型应用 面试教程 2026 版

## 开发环境

```bash
git clone https://github.com/BeefWrap4/LLM-Knowledge-Base.git
cd LLM-Knowledge-Base

# 安装 pre-commit (自动检查 wiki 链接 + 代码风格)
pip install pre-commit ruff
pre-commit install
```

## CI 流程

每次 push 都会自动跑 3 步验证 (~10 min):

| 步骤 | 检查 | 耗时 |
|------|------|------|
| 1. Quick check | wiki 链接 + 29/29 章节 README + smoke tests | <30s |
| 2. Core tier | 跑完 158 个 core/ 例子 | ~30s |
| 3. LLM tier | 跑完 199 个 llm/ 例子 (mock 模式) | ~3 min |

本地复现 CI:

```bash
cd code/
make install-llm
make ci                 # 完整 3 步
make ci-quick           # 仅 quick check (开发循环用)
make ci-core            # 仅 core tier
make ci-llm             # 仅 llm tier
```

## GPU tier 验证 (可选)

GPU 例子 (76 个) 需要 NVIDIA GPU + 重型依赖 (vllm/peft/mlx-lm) — 在 self-hosted runner 上手动触发:

```bash
# 手动触发 GitHub Actions
gh workflow run gpu-verify.yml

# 本地 (需 NVIDIA GPU)
cd code/
make install-gpu
make run-all TIER=gpu PARALLEL=1
```

## 添加新章节

1. 新建 `NN_TopicName.md` (NN = 两位数字, 零填充)
2. 复制已有章节的 frontmatter, 更新 `chapter` / `topic` / `difficulty` / `interview_frequency`
3. 包含 8 个必需段: 引言块 + 编号章节 + Mermaid 图 + Python 代码 + LaTeX + `## 📋 本章速查表` + `## 🎯 面试真题精讲` + `## 📚 相关章节`
4. 添加 `[[WikiLinks]]` (不是 markdown 链接) 到其他章节
5. **必须通过** `make ci-quick` 才算完成
6. **强烈建议** 同时在 `code/chNN_topicname/` 添加 1-2 个对照代码例子

## 添加新代码例子

每个 `.py` 文件必须有 YAML 风格 `# ---` frontmatter (不破坏 `python file.py` 可执行性):

```python
# ---
# chapter: 12
# topic: Scaled Dot-Product Attention
# section: 12.2.5
# difficulty: ⭐⭐⭐⭐
# tier: core           # core | llm | gpu
# deps: torch          # pip 依赖
# run: python 01_attention.py
# expected_runtime: <5s
# expected_output: 注意力权重 shape
# ---
#
# See: ../tutorial/12_Transformer与大模型原理.md §12.2.5
# Cross-refs: §12.3.2 MHA, Ch16.4 Flash Attention
# Interview hooks: "为什么除以 sqrt(d_k)?"

import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    ...

if __name__ == "__main__":
    # smoke test
    ...
    print("OK")
```

`SKIP` 模式 (deps 缺失时优雅跳过):

```python
try:
    import xgboost
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

if not HAS_XGB:
    print(f"[SKIP] xgboost not installed")
    print("OK")
    import sys; sys.exit(0)
```

## 常见错误

- ❌ Mermaid `mindmap` 块 — 改为 Unicode 文本树 (`├── └── │`)
- ❌ Markdown 链接 — 改用 `[[WikiLinks]]`
- ❌ 章节编号跳号 — 保持顺序
- ❌ `📚 相关章节` 用 `###` 或 inline `>` — 必须用 `##` (h2)
- ❌ 速查表缺 emoji — 必须用 `## 📋 本章速查表`
- ❌ 代码例子不能 `python file.py` 跑通 — 必须有 `if __name__ == "__main__":` + `print("OK")`
