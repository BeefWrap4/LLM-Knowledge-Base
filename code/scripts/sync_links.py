#!/usr/bin/env python3
# ---
# code/scripts/sync_links.py
# 教程 ↔ code 双向链接同步工具
# Usage:
#   python code/scripts/sync_links.py              # 报告
#   python code/scripts/sync_links.py --inject     # 自动给教程补充 → [code] 链接
# ---
"""
解析教程 ## X.Y(.Z...) 章节 + code # section: X.Y(.Z...), 验证双向链接.

报告 3 项:
  1. 教程章节覆盖: 哪些 §X.Y 有 code 例子, 哪些没有
  2. Code 例子有效性: 哪些 # section: 引用了真实教程章节
  3. 失效引用: 列出所有指向不存在章节的 code 文件

支持 --inject 模式: 自动给教程章节末尾添加 "→ [code: ...]" 链接 (如果该章节有 code).
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TUTORIAL_DIR = REPO
CODE_DIR = REPO / "code"

# 支持 12.2、12.2.5、24.6.5.1 等任意合理深度，避免四级编号被静默忽略。
TUTORIAL_SECTION_RE = re.compile(
    r"^#{2,6}\s+(\d{1,2}(?:\.\d+)+)\s+(.+?)(?:\s+⭐+)?\s*$"
)
# 匹配 code frontmatter 中的 section: X.Y(.Z...) + 可选标题
CODE_SECTION_RE = re.compile(r"#\s*section:\s*(\d{1,2}(?:\.\d+)+)(?:\s+(.+?))?\s*$")


def parse_tutorial_sections() -> dict[tuple[int, ...], list[tuple[Path, int, str]]]:
    """Returns {(chapter, section, ...): [(file, line, title), ...]}.

    Example: {(12, 2, 5): [(Path('12_...md'), 123, 'Self-Attention ...')]}
    """
    sections: dict[tuple[int, ...], list[tuple[Path, int, str]]] = defaultdict(list)
    for md in sorted(TUTORIAL_DIR.glob("[0-9][0-9]_*.md")):
        if md.name in (
            "00_目录索引.md",
            "99_库健康检查报告.md",
            "README.md",
            "CLAUDE.md",
            "CONTRIBUTING.md",
        ):
            continue
        for line_no, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            m = TUTORIAL_SECTION_RE.match(line)
            if m:
                number, title = m.groups()
                key = tuple(int(part) for part in number.split("."))
                sections[key].append((md, line_no, title.strip()))
    return sections


def parse_code_sections() -> list[tuple[Path, tuple[int, ...], str]]:
    """Returns [(code_file, (ch, section, ...), title), ...]."""
    results: list[tuple[Path, tuple, str]] = []
    for py in sorted(CODE_DIR.glob("ch*/*/*.py")):
        for line in py.read_text(encoding="utf-8").splitlines()[:15]:
            m = CODE_SECTION_RE.match(line)
            if m:
                number, title = m.groups()
                key = tuple(int(part) for part in number.split("."))
                results.append((py, key, (title or "").strip()))
                break
    return results


def normalize_section(key: tuple[int, ...]) -> tuple[int, int]:
    """Normalize any section depth to its chapter + main-section key."""
    return key[0], key[1]


def coverage_report(tut_sections: dict, code_refs: list) -> tuple[int, int, list, list]:
    """Returns (total_sections, sections_with_code, orphan_code_refs, missing_sections).

    Smart matching: if code refs §1.1.2 and tutorial has only §1.1, treat as match.
    """
    # All exact tutorial keys and their chapter + main-section parents.
    tut_exact_keys = set(tut_sections)
    tut_main_keys = {normalize_section(key) for key in tut_sections}

    # Code references: smart matching
    code_keys = set()
    orphan = []
    for py, key, title in code_refs:
        if key in tut_exact_keys:
            code_keys.add(key)
        elif normalize_section(key) in tut_main_keys:
            code_keys.add(normalize_section(key))
        else:
            orphan.append((py, key, title))

    # Sections with code
    sections_with_code = set()
    for k in code_keys:
        sections_with_code.add(normalize_section(k))

    # Tutorial sections without any code
    missing = []
    for key in tut_sections.keys():
        if normalize_section(key) not in sections_with_code:
            missing.append(key)

    return len(tut_sections), len(sections_with_code), orphan, missing


def inject_links(tut_sections: dict, code_refs: list) -> dict[Path, list[str]]:
    """Generate inline "→ [code: file.py]" links to inject under each tutorial section.

    Returns: {tutorial_md: [new_line_to_append, ...]}
    """
    # Build (ch, sec, sub) -> [(code_path_rel, title), ...]
    code_by_section: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for py, key, title in code_refs:
        rel_path = str(py.relative_to(REPO))
        code_by_section[key].append((rel_path, title))

    # Group by tutorial md
    injects: dict[Path, list[tuple[int, str]]] = defaultdict(list)
    for key, refs in tut_sections.items():
        # Find code for this section
        # Match the exact section first, then the chapter + main-section parent.
        candidates = list(code_by_section.get(key, []))
        parent = normalize_section(key)
        if key != parent:
            candidates += code_by_section.get(parent, [])
        if not candidates:
            continue
        # Find the first reference's md + line
        md, line_no, section_title = refs[0]
        # Build link line
        unique_files = sorted(set(rel for rel, _ in candidates))
        if len(unique_files) == 1:
            rel, t = unique_files[0]
            link_line = f"> → **代码示例**: [`{rel}`]({rel})"
            if t:
                link_line += f" — {t}"
        else:
            lines = [f"> → **代码示例** ({len(unique_files)} 个):"]
            for rel in unique_files:
                lines.append(f">   - [`{rel}`]({rel})")
            link_line = "\n".join(lines)
        injects[md].append((line_no, link_line))

    return injects


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", action="store_true", help="自动给教程补充 → [code] 链接")
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=13.0,
        help="章节主节代码覆盖率最低百分比；默认守住当前 13%% 基线",
    )
    args = ap.parse_args()

    tut_sections = parse_tutorial_sections()
    code_refs = parse_code_sections()

    print("=" * 60)
    print("  TUTORIAL ↔ CODE BIDIRECTIONAL LINK SYNC")
    print("=" * 60)

    print(f"\n[1/3] 教程章节总数: {len(tut_sections)}")
    by_chapter = defaultdict(int)
    for key in tut_sections:
        by_chapter[key[0]] += 1
    for ch in sorted(by_chapter):
        print(f"  Ch{ch:02d}: {by_chapter[ch]} sections")

    print(f"\n[2/3] Code 例子含 # section: 引用: {len(code_refs)}")
    by_tier = defaultdict(int)
    for py, _, _ in code_refs:
        tier = py.parent.name
        by_tier[tier] += 1
    for tier, count in sorted(by_tier.items()):
        print(f"  {tier}/: {count} files")

    total_sections, sections_with_code, orphans, missing = coverage_report(tut_sections, code_refs)

    print("\n[3/3] 双向覆盖率")
    cov = 0.0
    if total_sections > 0:
        cov = sections_with_code * 100 / total_sections
        print(f"  教程章节有 code 覆盖: {sections_with_code}/{total_sections} ({cov:.1f}%)")

    if orphans:
        print(f"\n--- ORPHAN CODE REFS ({len(orphans)}) ---")
        for py, key, title in orphans[:20]:
            rel = str(py.relative_to(REPO))
            print(f"  {rel} → §{'.'.join(map(str, key))}  ({title or 'no title'})")
        if len(orphans) > 20:
            print(f"  ... and {len(orphans) - 20} more")

    if missing:
        print(f"\n--- TUTORIAL SECTIONS WITHOUT CODE ({len(missing)}) ---")
        for key in sorted(missing)[:20]:
            label = "§" + ".".join(map(str, key))
            print(f"  {label}  (in {tut_sections[key][0][0].name})")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    coverage_ok = cov >= args.min_coverage
    failures = []
    if orphans:
        failures.append("orphan code refs")
    if not coverage_ok:
        failures.append(f"coverage {cov:.1f}% < {args.min_coverage:.1f}%")
    print(f"\n=== {'PASS' if not failures else 'FAIL (' + '; '.join(failures) + ')'} ===")

    if args.inject and not failures:
        # Auto-inject links
        print("\n=== Injecting → [code] links ===")
        injects = inject_links(tut_sections, code_refs)
        for md, lines in injects.items():
            print(f"  {md.name}: {len(lines)} link(s) to inject (DRY RUN)")
        print("\n  Re-run with --write to actually modify files")
    elif args.inject:
        print("\n  ! Skip inject due to orphan code refs (fix those first)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
