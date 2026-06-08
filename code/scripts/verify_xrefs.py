#!/usr/bin/env python3
# ---
# code/scripts/verify_xrefs.py
# 验证教程中所有 [[WikiLinks]] 都指向真实存在的 .md 文件
# Usage: python code/scripts/verify_xrefs.py
# Exit code: 0 if all OK, 1 if broken links found
# ---
"""
扫描仓库根目录所有 .md 文件, 提取 [[WikiLinks]], 验证目标 .md 存在.

排除:
  - CLAUDE.md 中的占位符 (NN_TopicName, WikiLinks, 章节名)
  - 误判的 Python list 文字 (包含逗号的短内容)
  - 代码块 (```...```) 内的内容 (虽然 Obsidian 也解析, 但减少假阳)
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # code/scripts/ -> repo root
META_LINKS = {"NN_TopicName", "WikiLinks", "章节名", "filename"}


def extract_wiki_links(text: str) -> list[str]:
    """Extract [[...]] where content is a plausible wiki link target.
    A wiki link target must:
      - contain at least one Chinese character or letter
      - not contain commas, brackets, or parens (those are list literals)
      - not be in META_LINKS
    """
    # 简单模式: 内容只能包含中文/英文/数字/下划线
    pattern = re.compile(r"\[\[([\w一-鿿]+)\]\]")
    links = []
    for m in pattern.finditer(text):
        target = m.group(1)
        if target in META_LINKS:
            continue
        links.append(target)
    return links


def scan_repo(repo: Path) -> tuple[dict[str, list[tuple[str, str]]], list[Path]]:
    """Returns (all_links, md_files).

    all_links: { target: [(source_file, line_no), ...] }
    """
    md_files = sorted(repo.glob("*.md"))
    all_links: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for md in md_files:
        # Skip code blocks
        text = md.read_text(encoding="utf-8")
        in_code = False
        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            for target in extract_wiki_links(line):
                all_links[target].append((md.name, line_no))

    return dict(all_links), md_files


def main() -> int:
    all_links, md_files = scan_repo(REPO)

    # Categorize
    resolved: list[tuple[str, list[tuple[str, str]]]] = []
    broken: list[tuple[str, list[tuple[str, str]]]] = []
    for target, refs in sorted(all_links.items()):
        if (REPO / f"{target}.md").exists():
            resolved.append((target, refs))
        else:
            broken.append((target, refs))

    # Print
    print("=== Tutorial Cross-Reference Verification ===\n")
    print(f"Repo: {REPO}")
    print(f"Total .md files: {len(md_files)}")
    print(f"Unique wiki link targets: {len(all_links)}")
    print(f"Resolved: {len(resolved)}")
    print(f"Broken:   {len(broken)}\n")

    if broken:
        print("--- BROKEN LINKS ---")
        for target, refs in broken:
            print(f"\n  [[{target}]]  ({len(refs)} reference(s))")
            for src, ln in refs[:5]:
                print(f"    - {src}:{ln}")
            if len(refs) > 5:
                print(f"    ... and {len(refs) - 5} more")
        print()

    # Stats per chapter
    print("--- Chapter reference counts (top 10) ---")
    chapter_refs = defaultdict(int)
    for target, refs in all_links.items():
        prefix = target.split("_")[0]
        if prefix.isdigit():
            chapter_refs[prefix] += len(refs)
    for ch, count in sorted(chapter_refs.items(), key=lambda x: -x[1])[:10]:
        print(f"  Ch{ch}: {count} inbound refs")

    print(f"\n=== {'PASS' if not broken else 'FAIL'} ===")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
