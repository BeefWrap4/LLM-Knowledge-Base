#!/usr/bin/env python3
"""Validate tutorial-to-code ownership through stable topic IDs.

Section numbers describe reading order and are intentionally allowed to change. Runnable
examples therefore link to canonical chapters with ``topic_id`` + ``chapter`` metadata and
``code/TOPIC_MANIFEST.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CODE_DIR = REPO / "code"
MANIFEST = CODE_DIR / "TOPIC_MANIFEST.json"
EXPECTED_CHAPTERS = 54
EXPECTED_EXAMPLES = 433

# Kept as public parsing contracts for historical migration tools and regression tests.
TUTORIAL_SECTION_RE = re.compile(r"^#{2,6}\s+(\d{1,2}(?:\.\d+)+)\s+(.+?)(?:\s+⭐+)?\s*$")
CODE_SECTION_RE = re.compile(r"#\s*section:\s*(\d{1,2}(?:\.\d+)+)(?:\s+(.+?))?\s*$")
TOPIC_ID_RE = re.compile(r"(?m)^# topic_id:\s*([a-z0-9_.-]+)\s*$")
CHAPTER_RE = re.compile(r"(?m)^# chapter:\s*(\d+)\s*$")


def canonical_chapters() -> dict[int, Path]:
    return {
        int(path.name[:2]): path
        for path in sorted(REPO.glob("[0-9][0-9]_*.md"))
        if 1 <= int(path.name[:2]) <= EXPECTED_CHAPTERS
    }


def parse_tutorial_sections() -> dict[tuple[int, ...], list[tuple[Path, int, str]]]:
    sections: dict[tuple[int, ...], list[tuple[Path, int, str]]] = defaultdict(list)
    for chapter in canonical_chapters().values():
        for line_no, line in enumerate(chapter.read_text(encoding="utf-8").splitlines(), 1):
            match = TUTORIAL_SECTION_RE.match(line)
            if match:
                number, title = match.groups()
                sections[tuple(int(part) for part in number.split("."))].append(
                    (chapter, line_no, title.strip())
                )
    return sections


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        raise ValueError("code/TOPIC_MANIFEST.json is missing")
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported topic manifest schema")
    return value


def validate_manifest(manifest: dict, chapters: dict[int, Path]) -> list[str]:
    failures: list[str] = []
    entries = manifest.get("examples", [])
    topic_ids: set[str] = set()
    paths: set[str] = set()
    for entry in entries:
        topic_id = entry.get("topic_id", "")
        relative = entry.get("path", "")
        chapter = entry.get("chapter")
        if topic_id in topic_ids:
            failures.append(f"duplicate topic_id: {topic_id}")
        topic_ids.add(topic_id)
        if relative in paths:
            failures.append(f"duplicate example path: {relative}")
        paths.add(relative)
        path = CODE_DIR / relative
        if not path.is_file():
            failures.append(f"missing example: {relative}")
            continue
        if chapter not in chapters:
            failures.append(f"{relative}: invalid canonical chapter {chapter}")
            continue
        if entry.get("chapter_file") != chapters[chapter].name:
            failures.append(f"{relative}: stale chapter_file")
        text = path.read_text(encoding="utf-8")
        topic_match = TOPIC_ID_RE.search(text)
        chapter_match = CHAPTER_RE.search(text)
        if not topic_match or topic_match.group(1) != topic_id:
            failures.append(f"{relative}: topic_id metadata mismatch")
        if not chapter_match or int(chapter_match.group(1)) != chapter:
            failures.append(f"{relative}: chapter metadata mismatch")
        expected_see = f"# See: ../../../{chapters[chapter].name}"
        if expected_see not in text:
            failures.append(f"{relative}: canonical See link mismatch")

    actual_paths = {path.relative_to(CODE_DIR).as_posix() for path in CODE_DIR.glob("ch[0-9][0-9]_*/*/*.py")}
    for missing in sorted(actual_paths - paths):
        failures.append(f"manifest does not cover {missing}")
    for stale in sorted(paths - actual_paths):
        failures.append(f"manifest contains stale path {stale}")
    if manifest.get("chapter_count") != EXPECTED_CHAPTERS:
        failures.append("manifest chapter_count is stale")
    if manifest.get("example_count") != EXPECTED_EXAMPLES or len(entries) != EXPECTED_EXAMPLES:
        failures.append("manifest example_count is stale")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=50.0,
        help="minimum percentage of canonical chapters with at least one runnable example",
    )
    args = parser.parse_args()
    print("=" * 60)
    print("  TUTORIAL ↔ CODE TOPIC MANIFEST")
    print("=" * 60)
    chapters = canonical_chapters()
    sections = parse_tutorial_sections()
    try:
        manifest = load_manifest()
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"\n=== FAIL ({exc}) ===")
        return 1
    failures = validate_manifest(manifest, chapters)
    entries = manifest.get("examples", [])
    covered = sorted({entry["chapter"] for entry in entries if entry.get("chapter") in chapters})
    coverage = 100 * len(covered) / len(chapters) if chapters else 0.0
    print(f"\n[1/3] 教程章节总数: {len(chapters)}; 正文编号节: {len(sections)}")
    print(f"[2/3] Code 例子含 topic_id 引用: {len(entries)}")
    print(f"[3/3] 教程章节有 code 覆盖: {len(covered)}/{len(chapters)} ({coverage:.1f}%)")
    if coverage < args.min_coverage:
        failures.append(f"chapter coverage {coverage:.1f}% < {args.min_coverage:.1f}%")
    for failure in failures[:30]:
        print(f"  [FAIL] {failure}")
    if len(failures) > 30:
        print(f"  ... and {len(failures) - 30} more")
    print(f"\n=== {'PASS' if not failures else 'FAIL'} ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
