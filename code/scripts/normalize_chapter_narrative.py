#!/usr/bin/env python3
"""Normalize the 54 canonical chapters without changing their technical body content.

Usage:
    python code/scripts/normalize_chapter_narrative.py
    python code/scripts/normalize_chapter_narrative.py --check
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
REPO = CODE.parent
UPDATED_AT = "2026-08-05T00:00:00.000Z"
EXPECTED_CHAPTERS = 54

PARTS = (
    ("第一部分 Python 与后端工程基础", 1, 10),
    ("第二部分 机器学习与大模型基础", 11, 16),
    ("第三部分 Prompt、Context 与 RAG", 17, 21),
    ("第四部分 Agent 与工程框架", 22, 28),
    ("第五部分 数据、训练、对齐、评估与安全", 29, 39),
    ("第六部分 推理服务与 LLMOps", 40, 46),
    ("第七部分 多模态与前沿架构", 47, 53),
    ("第八部分 岗位与项目面试实战", 54, 54),
)

TOPIC_IDS = {
    1: "python-runtime",
    2: "python-object-model",
    3: "python-functions-scope-decorators",
    4: "python-iteration-functional",
    5: "python-oop-data-model",
    6: "python-memory-profiling",
    7: "python-concurrency",
    8: "data-structures-algorithms",
    9: "numpy-pandas",
    10: "fastapi-backend",
    11: "machine-learning",
    12: "deep-learning-pytorch",
    13: "tokenizer-vocabulary",
    14: "attention-math-shapes",
    15: "transformer-architecture",
    16: "llm-pretraining-decoding-selection",
    17: "prompt-engineering",
    18: "context-engineering",
    19: "rag-ingestion-indexing",
    20: "rag-retrieval-reranking",
    21: "production-rag",
    22: "agent-tools",
    23: "mcp-a2a-skills",
    24: "agent-workflow-multi-agent",
    25: "durable-agent-runtime",
    26: "agent-memory-personalization",
    27: "llm-framework-selection",
    28: "computer-use-gui-agent",
    29: "llm-data-engineering",
    30: "sft-lora-qlora",
    31: "preference-alignment-rl",
    32: "reasoning-test-time-compute",
    33: "distributed-training",
    34: "jax-xla-tpu",
    35: "training-stability",
    36: "llm-evaluation-foundations",
    37: "rag-agent-safety-evaluation",
    38: "llm-agent-security",
    39: "ai-privacy-ethics-governance",
    40: "inference-memory-quantization-batching",
    41: "inference-engines-serving",
    42: "pd-disaggregation-kv-pooling",
    43: "cloud-native-model-gateway",
    44: "llmops-lifecycle-delivery",
    45: "llm-observability-sre",
    46: "edge-browser-llm",
    47: "multimodal-representation-llm",
    48: "diffusion-generative-vision",
    49: "world-models-vla-embodied",
    50: "ssm-mamba",
    51: "moe-mla-mtp-deepseek",
    52: "knowledge-editing-unlearning",
    53: "model-merging",
    54: "china-llm-interview",
}

RELATED_CODE = {
    3: ("ch01_python_runtime", "ch04_iteration_functional"),
    14: ("ch15_transformer",),
    16: ("ch15_transformer",),
    20: ("ch19_rag_indexing",),
    21: ("ch19_rag_indexing", "ch47_multimodal"),
    23: ("ch22_agent_tools",),
    24: ("ch22_agent_tools", "ch27_llm_frameworks"),
    25: ("ch22_agent_tools", "ch43_cloudnative"),
    26: ("ch22_agent_tools", "ch18_context_engineering"),
    28: ("ch17_prompt_engineering",),
    31: ("ch30_lora_qlora",),
    37: ("ch36_evaluation", "ch19_rag_indexing"),
    39: ("ch38_safety",),
    40: ("ch30_lora_qlora", "ch41_inference_engines"),
    42: ("ch41_inference_engines",),
    45: ("ch44_llmops",),
    48: ("ch47_multimodal",),
    51: ("ch15_transformer",),
    54: (),
}


def canonical_chapters() -> list[Path]:
    return [
        path for path in sorted(REPO.glob("[0-9][0-9]_*.md")) if 1 <= int(path.name[:2]) <= EXPECTED_CHAPTERS
    ]


def part_for(number: int) -> str:
    return next(name for name, start, end in PARTS if start <= number <= end)


def chapter_title(path: Path, text: str) -> str:
    number = int(path.name[:2])
    match = re.search(rf"(?m)^# 第 {number} 章 (.+?)(?:\s+⭐+)?$", text)
    if not match:
        raise ValueError(f"{path.name}: missing canonical H1")
    return match.group(1).strip()


def _outside_fence_heading_indices(lines: list[str], level: int) -> list[int]:
    indices: list[int] = []
    fence: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        marker_match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence is not None:
            if re.match(rf"^\s{{0,3}}{re.escape(fence[0])}{{{fence[1]},}}\s*$", line):
                fence = None
            continue
        if marker_match:
            marker = marker_match.group(1)
            fence = (marker[0], len(marker))
            continue
        if re.match(rf"^{'#' * level}\s+", line):
            indices.append(index)
    return indices


def _technical_body(text: str, number: int) -> list[str]:
    lines = text.splitlines()
    start = next(
        (
            index
            for index in _outside_fence_heading_indices(lines, 2)
            if re.match(rf"^## {number}\.\d+\b", lines[index])
        ),
        None,
    )
    end = next((index for index, line in enumerate(lines) if line == "## 🧭 本章小结"), None)
    if start is None or end is None or start >= end:
        raise ValueError(f"Ch{number:02d}: technical body or appendix boundary missing")
    return lines[start:end]


def _clean_heading(value: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)+\s+", "", value).strip()


def _renumber_body(lines: list[str], number: int) -> tuple[list[str], list[str]]:
    out: list[str] = []
    titles: list[str] = []
    fence: tuple[str, int] | None = None
    h2 = h3 = h4 = 0
    for line in lines:
        marker_match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence is not None:
            out.append(line)
            if re.match(rf"^\s{{0,3}}{re.escape(fence[0])}{{{fence[1]},}}\s*$", line):
                fence = None
            continue
        if marker_match:
            marker = marker_match.group(1)
            fence = (marker[0], len(marker))
            out.append(line)
            continue
        match2 = re.match(r"^##\s+(.+)$", line)
        if match2:
            h2 += 1
            h3 = h4 = 0
            title = _clean_heading(match2.group(1))
            titles.append(title)
            out.append(f"## {number}.{h2} {title}")
            continue
        match3 = re.match(r"^###\s+(.+)$", line)
        if match3:
            h3 += 1
            h4 = 0
            out.append(f"### {number}.{h2}.{h3} {_clean_heading(match3.group(1))}")
            continue
        match4 = re.match(r"^####\s+(.+)$", line)
        if match4:
            h4 += 1
            out.append(f"#### {number}.{h2}.{h3}.{h4} {_clean_heading(match4.group(1))}")
            continue
        out.append(line)
    return out, titles


def _chapter_stems() -> dict[int, str]:
    return {int(path.name[:2]): path.stem for path in canonical_chapters()}


def _wikilink(number: int, stems: dict[int, str], titles: dict[int, str]) -> str:
    return f"[[{stems[number]}|第 {number} 章 {titles[number]}]]"


def _code_dirs(number: int) -> tuple[str, ...]:
    direct = tuple(path.name for path in CODE.glob(f"ch{number:02d}_*") if path.is_dir())
    return direct or RELATED_CODE.get(number, ())


def _navigation(
    number: int, title: str, titles: list[str], stems: dict[int, str], all_titles: dict[int, str]
) -> str:
    previous = None if number == 1 else _wikilink(number - 1, stems, all_titles)
    prerequisites = "无；本章是全书起点。" if previous is None else f"{previous}。"
    selected = titles[:3] or [title]
    while len(selected) < 3:
        selected.append(selected[-1])
    path = " → ".join(titles[:6]) + (" → 生产边界与面试表达" if len(titles) > 6 else "")
    dirs = _code_dirs(number)
    code_line = "、".join(f"`code/{directory}/`" for directory in dirs) if dirs else "本章暂无独立代码目录"
    intro = (
        f"本章先回答“{titles[0] if titles else title}”为什么成立，再沿着机制、实现、评估和边界逐步展开。"
        "阅读时先建立因果链，再运行或推演示例，最后用章末自测检查能否脱离原文复述。"
    )
    return f"""> [!abstract] 本章导航
> **定位**：{part_for(number)}中的第 {number} 章；围绕“{title}”建立单一、可追踪的知识主线。
>
> **先修**：{prerequisites}
>
> **学习目标**：
> - 解释 {selected[0]} 的核心问题、机制与适用边界。
> - 实现或评估 {selected[1]} 的最小闭环。
> - 使用可复现证据诊断 {selected[2]} 的工程取舍与失败模式。
>
> **建议路径**：{path}。
>
> **配套代码**：{code_line}。

{intro}
"""


def _appendix(number: int, titles: list[str], stems: dict[int, str], all_titles: dict[int, str]) -> str:
    selected = titles[:5]
    summary = "\n".join(f"- {title}：能够说清问题、机制、证据与边界。" for title in selected[:3])
    tests = selected[:3] or [all_titles[number]]
    while len(tests) < 3:
        tests.append(tests[-1])
    dirs = _code_dirs(number)
    if dirs:
        listings = "\n".join(f"- `code/{directory}/`" for directory in dirs)
        commands = "\n".join(
            f"python code/scripts/run_all_examples.py --chapter ch{int(directory[2:4]):02d} --tier core"
            for directory in dirs
        )
        code = f"""{listings}

```powershell
{commands}
```

默认验收不下载模型、不调用付费 API；真实 API 或 GPU 示例必须按 metadata 显式启用。成功标准是相关脚本输出 `OK`，条件不足时输出可解释的 `[SKIP]`。"""
    else:
        code = "本章暂无独立代码目录。先完成正文中的设计题与自测；跨章示例以导航中指向的伴侣目录为准。"
    related = []
    if number > 1:
        related.append(_wikilink(number - 1, stems, all_titles))
    if number < EXPECTED_CHAPTERS:
        related.append(_wikilink(number + 1, stems, all_titles))
    related_lines = "\n".join(f"- {item}" for item in related)
    quick = "\n".join(f"| {title} | 问题 → 机制 → 示例 → 指标 → 边界 |" for title in selected)
    return f"""## 🧭 本章小结

{summary}

## ✅ 自测与练习

1. 不看正文，解释“{tests[0]}”解决什么问题，并给出一个不适用场景。
2. 为“{tests[1]}”设计一个最小可复现实验，明确输入、指标和通过条件。
3. 比较“{tests[2]}”的至少两种方案，说明质量、成本、延迟或风险取舍。

## 🧪 配套代码与验收

{code}

## 🎯 面试题精讲

回答本章问题时使用四步结构：先给结论，再解释机制，然后给项目证据，最后主动说明适用边界。涉及性能或效果时，补充模型、硬件、数据、并发、版本和统计口径；条件不完整时明确说“需要实测”。

## 📋 本章速查表

| 主题 | 回答主线 |
|---|---|
{quick}

## 🔗 相关章节

{related_lines}

## 📖 一手参考资料

> 核验基线：2026-07-31；结构复核：2026-08-05。产品、API、法规、价格与 benchmark 会变化，使用前应再次核验。

- [[docs/AUTHORITATIVE_SOURCES|章节权威来源索引]]：按主题维护官方文档、标准、原论文和官方仓库。
"""


def normalize_chapter(path: Path) -> str:
    chapters = canonical_chapters()
    stems = {int(item.name[:2]): item.stem for item in chapters}
    texts = {int(item.name[:2]): item.read_text(encoding="utf-8") for item in chapters}
    all_titles = {
        number: chapter_title(next(item for item in chapters if int(item.name[:2]) == number), text)
        for number, text in texts.items()
    }
    number = int(path.name[:2])
    text = texts[number]
    title = all_titles[number]
    body, titles = _renumber_body(_technical_body(text, number), number)
    frequency_match = re.search(rf"(?m)^# 第 {number} 章 .+?(\s+⭐+)$", text)
    stars = frequency_match.group(1).strip() if frequency_match else "⭐⭐⭐⭐"
    frontmatter = f"""---
chapter: {number}
topic: {title}
topic_id: {TOPIC_IDS[number]}
difficulty: 中高
interview_frequency: {len(stars)}
created: 2026-06-01T00:00:00.000Z
updated: {UPDATED_AT}
tags:
  - {TOPIC_IDS[number]}
  - 面试教程
---"""
    return "\n".join(
        (
            frontmatter,
            f"# 第 {number} 章 {title} {stars}",
            _navigation(number, title, titles, stems, all_titles).rstrip(),
            "\n".join(body).strip(),
            _appendix(number, titles, stems, all_titles).rstrip(),
            "",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    chapters = canonical_chapters()
    if len(chapters) != EXPECTED_CHAPTERS:
        print(f"[FAIL] chapters={len(chapters)}, expected={EXPECTED_CHAPTERS}")
        return 1
    changed: list[Path] = []
    for chapter in chapters:
        normalized = normalize_chapter(chapter)
        if normalized != chapter.read_text(encoding="utf-8"):
            changed.append(chapter)
            if not args.check:
                chapter.write_text(normalized, encoding="utf-8", newline="\n")
    if args.check and changed:
        for chapter in changed:
            print(f"[FAIL] not normalized: {chapter.name}")
        return 1
    print(f"[PASS] normalized chapters: {len(chapters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
