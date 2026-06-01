# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an **Obsidian vault** containing a 24-chapter interview preparation tutorial: "Python到大模型应用_面试教程_2026版". It covers the full stack from Python fundamentals to LLM deployment, targeting 2026 large-model algorithm/engineering interviews.

**26 files, ~1,387 KB total**. All files are markdown (`.md`) in a flat directory — no subdirectories.

## File Naming & Organization

```
NN_TopicName.md          where NN = two-digit chapter number
00_目录索引.md            Table of contents / navigation hub
01-06                    Python core (basics → concurrency → memory)
07-08                    Data structures & algorithms, data science
09                       Web dev (FastAPI)
10-11                    ML & DL fundamentals
12-16                    LLM core tech (Transformer, Prompt, RAG, Agent, Fine-tuning)
17-24                    LLM engineering practice (eval, frameworks, distributed, LLMOps, multimodal, data, safety, cloud)
99_库健康检查报告.md      Health audit report (fix records)
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

## Chapter Content Conventions

Each chapter follows this structure:
1. Opening quote block with difficulty/frequency summary
2. Numbered sections (e.g. `## 12.2 Self-Attention`) with ⭐ difficulty markers
3. Mermaid `flowchart`/`graph`/`sequenceDiagram` diagrams for architecture
4. Python code blocks with complete runnable examples
5. LaTeX math formulas in `$$` blocks
6. `## 本章思维导图` (Ch01-04) or `## 本章小结` (Ch05-08) — **text tree in code block**, NOT Mermaid mindmap
7. `## 🎯 面试真题精讲` — numbered interview questions with detailed answers
8. `## 📚 相关章节` — cross-references using `[[WikiLinks]]`

## Cross-References

- **Always use Obsidian wiki links**: `[[12_Transformer与大模型原理]]` — NOT markdown `[text](file.md)`
- Every chapter must have a `📚 相关章节` section with 3-5 links to related chapters
- Links must use the exact filename (minus `.md`) as the target

## Known Constraints

- **No Mermaid `mindmap` blocks** — all 9 were converted to Unicode text trees (`├── └── │`) in fenced code blocks. Mindmaps had `()` parsing bugs in Obsidian.
- **No unquoted `()` in any diagram node text**
- **No markdown links for cross-references** — always `[[wiki links]]`
- **Chapter section numbers must be sequential** (no gaps)

## Adding a New Chapter

1. Name: `NN_TopicName.md` (next available number, zero-padded)
2. Copy frontmatter from an existing chapter, update fields
3. Follow the content conventions above
4. Add the chapter to `00_目录索引.md` in the text tree and all index tables
5. Add [[wiki link]] references from 2-3 related existing chapters

## Obsidian MCP Tools

This vault is accessed via `obsidian_*` MCP tools:
- `obsidian_get_note` — read content, document-map, or specific sections
- `obsidian_write_note` / `obsidian_patch_note` — create or edit notes
- `obsidian_replace_in_note` — bulk find-and-replace (supports regex)
- `obsidian_search_notes` — full-text or JSONLogic search
- `obsidian_manage_frontmatter` — get/set/delete YAML fields
- `obsidian_manage_tags` — add/remove/list tags

Nested headings use `Parent::Child` syntax in section targeting.
