#!/usr/bin/env python3
"""Rebuild reader entry documents from the canonical 54-chapter catalog."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
REPO = CODE.parent
MANIFEST = CODE / "TOPIC_MANIFEST.json"

PARTS = (
    ("第一部分 Python 与后端工程基础", 1, 10, "建立能运行、能调试、能服务化的 Python 基座"),
    ("第二部分 机器学习与大模型基础", 11, 16, "从传统机器学习推进到 Tokenizer、Attention 与 Transformer"),
    ("第三部分 Prompt、Context 与 RAG", 17, 21, "构建外部知识增强和可评估的检索生成闭环"),
    ("第四部分 Agent 与工程框架", 22, 28, "把模型、工具、状态和人类审批组织成可恢复系统"),
    ("第五部分 数据、训练、对齐、评估与安全", 29, 39, "建立从数据到上线门禁的模型能力改造闭环"),
    ("第六部分 推理服务与 LLMOps", 40, 46, "优化数据面并建设可交付、可观测的生产系统"),
    ("第七部分 多模态与前沿架构", 47, 53, "理解多模态、具身和非标准架构的证据边界"),
    ("第八部分 岗位与项目面试实战", 54, 54, "把知识转成可验证项目证据和系统设计表达"),
)


def chapters() -> dict[int, Path]:
    result = {
        int(path.name[:2]): path
        for path in sorted(REPO.glob("[0-9][0-9]_*.md"))
        if 1 <= int(path.name[:2]) <= 54
    }
    if list(result) != list(range(1, 55)):
        raise ValueError("canonical Ch01-Ch54 files are incomplete")
    return result


def title(number: int, path: Path) -> str:
    match = re.search(rf"(?m)^# 第 {number} 章 (.+?)(?:\s+⭐+)?$", path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"missing H1: {path.name}")
    return match.group(1).strip()


def main_sections(number: int, path: Path) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(rf"^## {number}\.\d+\s+(.+?)(?:\s+⭐+)?$", line)
        if match:
            values.append(match.group(1).strip())
    return values


def code_counts() -> Counter[int]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return Counter(entry["chapter"] for entry in manifest["examples"])


def wikilink(number: int, files: dict[int, Path], titles: dict[int, str]) -> str:
    # Keep table cells free of the WikiLink alias pipe, which Markdown interprets as a column.
    return f"[[{files[number].stem}]]"


def build_index(
    files: dict[int, Path], titles: dict[int, str], sections: dict[int, list[str]], counts: Counter[int]
) -> str:
    blocks = []
    for part, start, end, outcome in PARTS:
        rows = []
        for number in range(start, end + 1):
            path = " → ".join(sections[number][:3])
            code = f"{counts[number]} 个" if counts[number] else "设计题/跨章示例"
            rows.append(f"| {number:02d} | {wikilink(number, files, titles)} | {path} | {code} |")
        blocks.append(
            f"""## {part}

> 学习产出：{outcome}。

| 章 | 主题 | 章内推进 | 代码 |
|---:|---|---|---:|
{chr(10).join(rows)}"""
        )
    return f"""# Python 到大模型应用：54 章目录索引

> [!abstract] 教程定位
> 这是一套面向 2026 年 Python、LLM 应用、训练与 AI Infra 岗位的中文教程。54 章按依赖关系组织，正文事实基线为 2026-07-31，分章结构复核于 2026-08-05。

| 指标 | 当前值 |
|---|---:|
| 教程章节 | 54 章 |
| 知识分部 | 8 大部分 |
| 代码伴侣目录 | 29 个 |
| 可运行示例 | 433 个 |
| 具备代码覆盖的规范章节 | {len(counts)}/54 |

```mermaid
flowchart LR
    P1["Python 与后端"] --> P2["ML 与 LLM 基础"]
    P2 --> P3["Prompt、Context 与 RAG"]
    P3 --> P4["Agent 与框架"]
    P2 --> P5["数据、训练、评估与安全"]
    P4 --> P6["推理服务与 LLMOps"]
    P5 --> P6
    P6 --> P7["多模态与前沿架构"]
    P7 --> P8["岗位与项目面试"]
```

图中的箭头表示主要先修关系，不表示只能线性阅读。应用工程读者可在完成 Ch01–Ch18 后进入 RAG/Agent 主线；训练与 Infra 读者应先完成 Ch11–Ch16。

{(chr(10) * 2).join(blocks)}

## 按目标选择学习路线

| 目标 | 建议顺序 | 验收产出 |
|---|---|---|
| LLM 应用 / RAG | Ch01–Ch18 → Ch19–Ch21 → Ch36–Ch37 → Ch43–Ch45 → Ch54 | 可回归 RAG、Golden Dataset、指标漏斗和上线门禁 |
| Agent 工程 | Ch17–Ch18 → Ch22–Ch28 → Ch37–Ch39 → Ch44–Ch45 → Ch54 | 可恢复状态机、幂等工具、副作用治理和轨迹评测 |
| Post-training | Ch11–Ch16 → Ch29–Ch35 → Ch36–Ch39 → Ch51–Ch53 | 数据配比、SFT/偏好优化、分布式训练和稳定性诊断 |
| 推理 / AI Infra | Ch14–Ch16 → Ch33–Ch35 → Ch40–Ch46 → Ch51 | TTFT/TPOT、KV Cache、引擎压测、K8s 与可观测性 |
| 多模态 / 具身 | Ch11–Ch16 → Ch21 → Ch47–Ch49 → Ch36–Ch39 | 多模态表征、生成、VLA 数据与安全评估 |

## 16 周建议节奏

| 周期 | 范围 | 通过条件 |
|---|---|---|
| 第 1–3 周 | Ch01–Ch10 | 能独立编写、测试并服务化 Python 程序 |
| 第 4–5 周 | Ch11–Ch16 | 能跟踪张量形状并解释训练与解码数据流 |
| 第 6–7 周 | Ch17–Ch21 | 完成一个可评估、可追踪来源的 RAG 最小系统 |
| 第 8–9 周 | Ch22–Ch28 | 完成一个可恢复、有权限边界的 Agent 工作流 |
| 第 10–12 周 | Ch29–Ch39 | 能设计数据、训练、评估和安全门禁 |
| 第 13–14 周 | Ch40–Ch46 | 能定位推理性能和线上可靠性问题 |
| 第 15 周 | Ch47–Ch53 | 能区分论文机制、产品能力与工程成熟度 |
| 第 16 周 | Ch54 | 完成项目陈述、系统设计和模拟面试 |

## 维护入口

- [[docs/RECHAPTERING_MAP|54 章重构迁移表]]
- [[docs/CHAPTER_STYLE_GUIDE|章节叙事与格式规范]]
- [[docs/AUTHORITATIVE_SOURCES|章节权威来源索引]]
- [`code/TOPIC_MANIFEST.json`](code/TOPIC_MANIFEST.json)：稳定主题 ID 与代码归属
- [[99_库健康检查报告|库健康检查报告]]
"""


def build_readme(files: dict[int, Path], titles: dict[int, str], counts: Counter[int]) -> str:
    part_rows = "\n".join(
        f"| {part} | Ch{start:02d}–Ch{end:02d} | {outcome} |" for part, start, end, outcome in PARTS
    )
    tree = "\n".join(f"├── {start:02d}–{end:02d}  {part}" for part, start, end, _ in PARTS)
    return f"""# Python 到大模型应用面试教程（2026 版）

> 54 章节 · 8 大部分 · 433 个可运行代码示例 · 事实基线：2026-07-31 · 结构审校：2026-08-05

[![chapters](https://img.shields.io/badge/chapters-54-blue.svg)](00_目录索引.md)
[![examples](https://img.shields.io/badge/examples-433-success.svg)](code/README.md)
[![python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](code/README.md)

这套教程把 Python、模型原理、RAG、Agent、训练、推理服务和岗位面试组织成一条可验证的学习链。每章都有导航、导入段、连续编号正文、自测、代码验收、面试表达、速查表和一手来源入口。

## 从哪里开始

- 第一次阅读：打开 [[00_目录索引]]，按 16 周主线推进。
- 已有 Python 基础：从 [[11_机器学习基础]] 或 [[13_Tokenizer与词表工程]] 开始。
- 只做 LLM 应用：重点阅读 Ch17–Ch28、Ch36–Ch45 和 Ch54。
- 只做训练 / Infra：重点阅读 Ch11–Ch16、Ch29–Ch46、Ch51。

## 54 章结构

| 分部 | 范围 | 学习产出 |
|---|---:|---|
{part_rows}

```text
{tree}
```

完整章节表、路线和周计划见 [[00_目录索引]]；旧 40 章如何迁移见 [[docs/RECHAPTERING_MAP]]。

## 代码仓库

`code/` 中有 29 个运行分组、433 个示例，目前覆盖 {len(counts)}/54 个规范章节。目录编号便于批量运行；每个示例的永久身份由 `topic_id` 决定，规范归属见 [`code/TOPIC_MANIFEST.json`](code/TOPIC_MANIFEST.json)。

```powershell
cd code
python -m pip install -r requirements-core.txt
python scripts/verify_all.py
python -m pytest -q -m "not gpu and not slow"
python -m ruff check .
```

运行单个示例：

```powershell
python ch15_transformer/core/01_scaled_dot_product_attention.py
```

批量运行一个分组：

```powershell
python scripts/run_all_examples.py --chapter ch22 --tier llm
```

默认 LLM/GPU 验收使用 mock 或条件跳过，不下载模型、不读取 Key、不产生付费请求。真实 API、GPU、Docker、Redis、pgvector 和模型权重需按脚本 metadata 显式启用。

## 模型与本地数据

模型权重不进入 Git。教程模型默认位于仓库外：

```text
E:\\AI_Models\\Projects\\MyDocument\\Python到大模型应用_面试教程_2026版\\models
```

下载任何内容优先尝试迅雷；模型权重优先在 ModelScope 查找，没有时再使用 Hugging Face。不要提交 `.env`、API Key、本地缓存或模型文件。

## Obsidian 阅读

将仓库根目录作为 Vault 打开。正文只使用仓库门禁覆盖的 WikiLink、Markdown、MathJax 和 Mermaid 语法；如果修改图、表格或公式，请运行：

```powershell
python code/scripts/verify_all.py
```

该命令检查 WikiLink、章节契约、Markdown/Obsidian 渲染风险、Mermaid、代码路径、来源索引、快照一致性和离线 smoke。

## 质量与贡献

- 章节规范：[[docs/CHAPTER_STYLE_GUIDE]]
- 章节模板：[[docs/CHAPTER_TEMPLATE]]
- 权威来源：[[docs/AUTHORITATIVE_SOURCES]]
- 代码说明：[`code/README.md`](code/README.md)
- 贡献约定：[`CONTRIBUTING.md`](CONTRIBUTING.md)
- 健康报告：[[99_库健康检查报告]]

提交前至少运行 `make ci-quick`、相关 pytest 和 `make lint`。真实 API/GPU 结果要注明版本、硬件、数据、并发和统计口径。

## 版本记录

- **2026-08-05** — 全面重构为 54 章、8 大部分；按 H2/H3 语义迁移正文，引入稳定 `topic_id` 和代码主题清单。
- **2026-08-04** — 统一旧 40 章导航、正文编号与学习闭环，加入 Obsidian 和结构门禁。
- **2026-07-31** — 完成事实、API、安全与可运行性审校，建立权威来源索引。
"""


def build_code_readme(counts: Counter[int]) -> str:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tiers = Counter(entry["tier"] for entry in manifest["examples"])
    directories = sorted(path for path in CODE.glob("ch[0-9][0-9]_*") if path.is_dir())
    rows = []
    for directory in directories:
        entries = [entry for entry in manifest["examples"] if entry["path"].startswith(directory.name + "/")]
        chapter_set = sorted({entry["chapter"] for entry in entries})
        rows.append(
            f"| `{directory.name}` | {len(entries)} | "
            + ", ".join(f"Ch{number:02d}" for number in chapter_set)
            + " |"
        )
    return f"""# 可运行代码伴侣

> 54 章教程的 29 个代码运行分组：433 个示例（{tiers["core"]} core + {tiers["llm"]} llm + {tiers["gpu"]} gpu）

目录编号用于批量运行，不再等同于每个文件的永久主题归属。所有示例都有唯一 `topic_id`；[`TOPIC_MANIFEST.json`](TOPIC_MANIFEST.json) 记录其规范章节、路径和 tier。

## 快速验收

```powershell
python -m pip install -r requirements-core.txt
make ci-quick
make test
make lint
```

没有 API Key 时：

```powershell
$env:LLM_MOCK = "1"
make test-llm
```

运行一个分组或单文件：

```powershell
python scripts/run_all_examples.py --chapter ch22 --tier llm
python ch15_transformer/core/01_scaled_dot_product_attention.py
```

## 运行分组

| 目录 | 示例 | 规范章节 |
|---|---:|---|
{chr(10).join(rows)}

## 示例契约

- Python 3.10+，四空格，Ruff 行宽 110。
- 文件开头包含 `# ---` metadata、`chapter`、稳定 `topic_id`、tier、依赖和预期结果。
- 脚本可直接运行，保留 `if __name__ == "__main__"`，成功输出 `OK`。
- 缺少可选依赖、真实 API 或 GPU 时输出明确 `[SKIP]`，不得伪造成功。
- 模型权重、缓存、`.env` 和 API Key 不得提交。

## 主题覆盖

当前 433 个示例覆盖 {len(counts)}/54 个规范章节。没有独立示例的专题章通过设计题、跨章示例或相邻运行分组验收，不为凑覆盖率复制代码。
"""


def build_mindmap(files: dict[int, Path], titles: dict[int, str], sections: dict[int, list[str]]) -> str:
    blocks = []
    for part, start, end, outcome in PARTS:
        chapter_lines = []
        for number in range(start, end + 1):
            section_list = "；".join(sections[number][:4])
            chapter_lines.append(f"  - Ch{number:02d} {titles[number]}\n    - {section_list}")
        blocks.append(f"- {part}\n  - 学习产出：{outcome}\n" + "\n".join(chapter_lines))
    return f"""# Python 到大模型应用面试教程：知识树

> 覆盖范围：54 章教程、8 大部分、433 个配套代码示例、CI 与 Docker 工程入口。事实基线 2026-07-31，结构版本 2026-08-05。

{chr(10).join(blocks)}

## 主干依赖

Python 工程 → 机器学习与张量 → Tokenizer / Attention / Transformer → Prompt / Context → RAG / Agent → 训练评估与安全 → 推理服务与 LLMOps → 多模态与前沿架构 → 项目面试证据。

## 使用方式

- 线性学习：按 Ch01–Ch54 阅读。
- 目标学习：使用 [[00_目录索引#按目标选择学习路线|目录中的岗位路线]]。
- 代码检索：使用 [`code/TOPIC_MANIFEST.json`](code/TOPIC_MANIFEST.json) 的稳定 `topic_id`。
- 内容维护：先查 [[docs/RECHAPTERING_MAP]] 的规范归属，避免跨章复制。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated entry documents are stale")
    args = parser.parse_args()
    files = chapters()
    titles = {number: title(number, path) for number, path in files.items()}
    sections = {number: main_sections(number, path) for number, path in files.items()}
    if any(not value for value in sections.values()):
        raise ValueError("every chapter must have at least one numbered H2")
    counts = code_counts()
    outputs = {
        REPO / "00_目录索引.md": build_index(files, titles, sections, counts),
        REPO / "README.md": build_readme(files, titles, counts),
        CODE / "README.md": build_code_readme(counts),
        REPO / "Python到大模型应用_面试教程_2026版_思维导图.xmind.md": build_mindmap(files, titles, sections),
    }
    stale = []
    for path, content in outputs.items():
        rendered = content.rstrip() + "\n"
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            stale.append(path)
            if not args.check:
                path.write_text(rendered, encoding="utf-8", newline="\n")
                print(f"[WRITE] {path.relative_to(REPO)}")
    if args.check and stale:
        for path in stale:
            print(f"[FAIL] stale generated document: {path.relative_to(REPO)}")
        return 1
    print(f"[PASS] catalog entry documents: {len(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
