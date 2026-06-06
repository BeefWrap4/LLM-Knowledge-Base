# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an **Obsidian vault** containing a 29-chapter interview preparation tutorial: "Python到大模型应用_面试教程_2026版". It covers the full stack from Python fundamentals to LLM deployment, targeting 2026 large-model algorithm/engineering interviews.

**33 files, ~2,200 KB total**. All markdown (`.md`) in a flat directory — no subdirectories.

- `README.md` — top-level overview for GitHub readers (岗位学习路径, 2026 主题速查, 快速开始)
- `00_目录索引.md` — Obsidian navigation hub (MOC, 板块架构, 速查索引)
- `01-24_*.md` — 24 core chapters
- `25-29_*.md` — 5 2026-era chapters (推理引擎, 世界模型, Test-Time Compute, 端侧 LLM, Context Engineering)
- `99_库健康检查报告.md` — health audit report (latest: 95/100)
- `CLAUDE.md` — this file

## File Naming & Organization

```
NN_TopicName.md          where NN = two-digit chapter number, zero-padded
00_目录索引.md            Table of contents / navigation hub
01-06                    Python core (basics → concurrency → memory)
07-08                    Data structures & algorithms, data science
09                       Web dev (FastAPI)
10-11                    ML & DL fundamentals
12-16                    LLM core tech (Transformer, Prompt, RAG, Agent, Fine-tuning)
17-24                    LLM engineering practice
25-29                    2026 new topics (inference engines, world models, TTC, edge, context)
99_库健康检查报告.md      Health audit report
README.md                 Top-level README (GitHub landing page)
```

## YAML Frontmatter Schema

Every chapter uses this frontmatter — keep it consistent:

```yaml
chapter: <number>
topic: <short Chinese title>
difficulty: 入门 | 中 | 中高 | 高 | 极高
interview_frequency: <number 0-5>
created: 2026-06-01T00:00:00.000Z   # ISO 8601
tags: [tag1, tag2, ...]
```

- `chapter` and `interview_frequency` must be **numbers**, not strings
- `difficulty` uses the five-tier Chinese system above
- `created` must be ISO 8601 format with time component
- For Ch00 (TOC), `difficulty` is `目录` and `interview_frequency` is `0` (not a real chapter)

## Chapter Content Conventions (100% Enforced)

Every chapter MUST contain all of these sections in order:

1. **Opening quote block** with difficulty/frequency summary (e.g. `> **面试频率**: ⭐⭐⭐⭐⭐`)
2. **Numbered sections** (`## 12.2 Self-Attention`) with ⭐ difficulty markers
3. **Mermaid diagrams** — `flowchart`/`graph`/`sequenceDiagram`/`timeline` for architecture
4. **Python code blocks** with complete runnable examples
5. **LaTeX math** in `$$` blocks
6. **`## 📋 本章速查表`** — Markdown table with 6-10 concept/keyword rows
7. **`## 🎯 面试真题精讲`** (or similar) — numbered interview questions with detailed answers
8. **`## 📚 相关章节`** — cross-references using `[[WikiLinks]]`

Plus:
- New 2026 chapters (Ch25-29) include a `## 章节小结` paragraph before the cheat sheet
- Chapters with depth ≥ 5 may have a `## 本章思维导图` (text tree) OR `## 本章小结` table — but currently ALL chapters use the standardized `## 📋 本章速查表` table at the end

## Cross-References

- **Always use Obsidian wiki links**: `[[12_Transformer与大模型原理]]` — NOT markdown `[text](file.md)`
- Every chapter must have a `📚 相关章节` section with 4-6 links to related chapters
- Each link should have a brief description of the relationship
- Links must use the exact filename (minus `.md`) as the target

## Known Constraints

- **No Mermaid `mindmap` blocks** — all were converted to Unicode text trees (`├── └── │`). Mindmaps have `()` parsing bugs in Obsidian. Also avoid `()` in flowchart/graph node text where possible.
- **No markdown links for cross-references** — always `[[wiki links]]`
- **Chapter section numbers must be sequential** (no gaps)
- **`📚 相关章节` heading must be `## ` (h2)**, not `### `, not inline `>` block
- **速查表 heading**: Use `## 📋 本章速查表` (not `## 本章速查表` without emoji, not `### `)

## Adding a New Chapter

1. Name: `NN_TopicName.md` (next available number, zero-padded)
2. Copy frontmatter from an existing chapter, update fields
3. Follow the content conventions above (all 8 required sections)
4. Add the chapter to `00_目录索引.md` in both the text tree and the 详细索引 table
5. Add 4-6 `[[wiki link]]` references FROM this chapter to others
6. Add `[[wiki link]]` references from 3-5 related existing chapters TO this chapter
7. Add the chapter to `99_库健康检查报告.md` Phase log
8. Commit with `git commit -m "Add NN [chapter-title]"`

## Obsidian MCP Tools

This vault is accessed via `obsidian_*` MCP tools (when available):
- `obsidian_get_note` — read content, document-map, or specific sections (use `format: "document-map"` for structure, `format: "section"` with `target` for specific heading)
- `obsidian_write_note` / `obsidian_patch_note` — create or edit notes
- `obsidian_replace_in_note` — bulk find-and-replace (supports regex)
- `obsidian_search_notes` — full-text or JSONLogic search
- `obsidian_manage_frontmatter` — get/set/delete YAML fields
- `obsidian_manage_tags` — add/remove/list tags

Nested headings use `Parent::Child` syntax in section targeting.

**When MCP tools are unavailable** (disconnected state), fall back to Read/Edit/Write/Glob with `D:\MyDocument\Python到大模型应用_面试教程_2026版_分章节\` paths.

## 2026 主题 Coverage (Audit Reference)

| 主题 | 章节 | 状态 |
|------|------|------|
| 推理引擎 (vLLM/SGLang/TensorRT-LLM) | Ch25 | ✅ |
| 世界模型 / VLA / LeRobot | Ch26 | ✅ |
| Reasoning Models (o3/R1) | Ch27 | ✅ |
| 端侧 LLM (Apple MLX) | Ch28 | ✅ |
| Context Engineering | Ch29 | ✅ |
| MCP / A2A / Skills | Ch15 | ✅ |
| RL Post-Training | Ch16.11 | ✅ |
| Langfuse v3 / Phoenix / OpenInference | Ch17.11, Ch20.10 | ✅ |
| Muon Optimizer | Ch19.9 | ✅ |
| Pydantic AI / Strands / OpenAI Agents | Ch18.8 | ✅ |

## Git Workflow

- Branch: `master` (main, only)
- Commit style: `Add N [description]` / `Polish: [description]` / `Update [target]: [description]`
- Commit per logical change, not per file
- Health audit: see `99_库健康检查报告.md` (target: ≥ 90/100)
