#!/usr/bin/env python3
"""Build and verify stable topic IDs for all runnable examples.

Chapter numbers describe the current learning order and may change. ``topic_id`` is the stable
identity used by documentation and maintenance tooling.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
REPO = CODE.parent
MANIFEST = CODE / "TOPIC_MANIFEST.json"


OVERRIDES: dict[str, list[tuple[int, str]]] = {
    "ch01_python_runtime": [
        (2, r"^(?:07_basic_data_types|11_|12_|13_|14_|15_|22_)"),
        (3, r"^(?:16_|17_|18_|19_)"),
    ],
    "ch04_iteration_functional": [(3, r"^(?:0[1-8]_|16_|17_)")],
    "ch15_transformer": [(14, r"^(?:01_|02_)"), (31, r"^05_")],
    "ch17_prompt_engineering": [
        (18, r"^(?:13_|14_|15_|21_|22_)"),
        (28, r"^(?:16_|17_)"),
        (32, r"^12_"),
        (38, r"^11_"),
    ],
    "ch18_context_engineering": [
        (26, r"^(?:06_|07_|08_|11_)"),
        (24, r"^09_"),
    ],
    "ch19_rag_indexing": [
        (20, r"^(?:11_|12_|13_|14_|15_)"),
        (21, r"^(?:16_|17_|18_|19_|20_|21_|22_|23_|24_)"),
        (37, r"^25_"),
    ],
    "ch22_agent_tools": [
        (23, r"^(?:04_|05_|15_|16_)"),
        (24, r"^08_"),
        (25, r"^(?:09_|20_)"),
        (26, r"^(?:06_|07_)"),
        (38, r"^(?:10_|11_|12_|13_|14_|19_|22_)"),
    ],
    "ch27_llm_frameworks": [
        (26, r"^(?:05_|06_|07_|09_|35_|36_)"),
        (24, r"^(?:12_|23_|24_|25_)"),
        (30, r"^(?:19_|20_)"),
    ],
    "ch29_data_engineering": [(32, r"^10_")],
    "ch30_lora_qlora": [
        (40, r"^(?:03_|04_|05_)"),
        (41, r"^(?:06_|07_)"),
        (46, r"^(?:08_|09_|10_|11_)"),
        (31, r"^(?:12_|13_|14_|15_)"),
    ],
    "ch36_evaluation": [(37, r"^(?:06_|07_|08_|09_|11_|12_|13_|14_)")],
    "ch38_safety": [(39, r"^(?:05_|06_|07_|08_|09_)"), (24, r"^14_")],
    "ch41_inference_engines": [
        (40, r"^(?:01_|02_|05_|06_|09_)"),
        (42, r"^04_"),
    ],
    "ch43_cloudnative": [(44, r"^04_"), (46, r"^05_")],
    "ch44_llmops": [(45, r"^(?:01_|05_|06_|07_|08_|11_|12_|13_|14_|15_|16_|19_|20_|21_|22_|23_)")],
    "ch47_multimodal": [(48, r"^05_"), (49, r"^(?:08_|09_|10_)"), (21, r"^11_")],
}


def chapter_files() -> dict[int, Path]:
    return {
        int(path.name[:2]): path
        for path in sorted(REPO.glob("[0-9][0-9]_*.md"))
        if 1 <= int(path.name[:2]) <= 54
    }


def chapter_titles(files: dict[int, Path]) -> dict[int, str]:
    result: dict[int, str] = {}
    for number, path in files.items():
        match = re.search(rf"(?m)^# 第 {number} 章 (.+?)(?:\s+⭐+)?$", path.read_text(encoding="utf-8"))
        if not match:
            raise ValueError(f"missing canonical H1: {path.name}")
        result[number] = match.group(1).strip()
    return result


def canonical_chapter(directory: str, stem: str) -> int:
    default = int(directory[2:4])
    for number, pattern in OVERRIDES.get(directory, []):
        if re.search(pattern, stem):
            return number
    return default


def stable_topic_id(directory: str, stem: str) -> str:
    domain = re.sub(r"^ch\d{2}_", "", directory)
    topic = re.sub(r"^\d+_", "", stem)
    return f"{domain}.{topic}".replace("-", "_")


def _replace_metadata(text: str, *, chapter: int, title: str, topic_id: str, chapter_file: str) -> str:
    text = re.sub(r"(?m)^# chapter:\s*(?:Ch)?\d+\s*$", f"# chapter: {chapter}", text, count=1)
    text = re.sub(r"(?m)^# topic:\s*.*$", f"# topic: {title}", text, count=1)
    if re.search(r"(?m)^# topic_id:", text):
        text = re.sub(r"(?m)^# topic_id:\s*.*$", f"# topic_id: {topic_id}", text, count=1)
    else:
        topic = re.search(r"(?m)^# topic:.*$", text)
        if not topic:
            raise ValueError("missing # topic metadata")
        text = text[: topic.end()] + f"\n# topic_id: {topic_id}" + text[topic.end() :]
    text = re.sub(r"(?m)^# section:\s*.*\n", "", text, count=1)
    see = f"# See: ../../../{chapter_file}"
    if re.search(r"(?m)^# See:", text):
        text = re.sub(r"(?m)^# See:.*$", see, text, count=1)
    else:
        metadata_end = text.find("# ---", text.find("# ---") + 1)
        if metadata_end == -1:
            raise ValueError("missing metadata closing marker")
        insert = text.find("\n", metadata_end) + 1
        text = text[:insert] + see + "\n" + text[insert:]
    return text


def build(write: bool) -> dict:
    files = chapter_files()
    if len(files) != 54:
        raise ValueError(f"expected 54 chapters, got {len(files)}")
    titles = chapter_titles(files)
    entries = []
    ids: Counter[str] = Counter()
    by_directory: dict[Path, list[dict]] = defaultdict(list)
    for path in sorted(CODE.glob("ch[0-9][0-9]_*/*/*.py")):
        directory = path.parent.parent
        chapter = canonical_chapter(directory.name, path.stem)
        topic_id = stable_topic_id(directory.name, path.stem)
        ids[topic_id] += 1
        entry = {
            "topic_id": topic_id,
            "chapter": chapter,
            "chapter_file": files[chapter].name,
            "path": path.relative_to(CODE).as_posix(),
            "tier": path.parent.name,
        }
        entries.append(entry)
        by_directory[directory].append(entry)
        if write:
            text = path.read_text(encoding="utf-8")
            updated = _replace_metadata(
                text,
                chapter=chapter,
                title=titles[chapter],
                topic_id=topic_id,
                chapter_file=files[chapter].name,
            )
            path.write_text(updated, encoding="utf-8", newline="\n")
    duplicates = sorted(topic_id for topic_id, count in ids.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate topic_id values: {duplicates}")
    manifest = {
        "schema_version": 1,
        "generated_at": "2026-08-05",
        "chapter_count": 54,
        "example_count": len(entries),
        "examples": entries,
    }
    if write:
        MANIFEST.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        for directory, directory_entries in by_directory.items():
            counts = Counter(entry["tier"] for entry in directory_entries)
            chapters = sorted({entry["chapter"] for entry in directory_entries})
            links = "\n".join(
                f"- [第 {chapter} 章 {titles[chapter]}](../../{files[chapter].name})" for chapter in chapters
            )
            tiers = "、".join(f"{tier}={counts[tier]}" for tier in sorted(counts))
            readme = f"""# {directory.name} 代码伴侣

> 本目录是运行分组，不是主题身份。示例的规范归属以 [`code/TOPIC_MANIFEST.json`](../TOPIC_MANIFEST.json) 中的稳定 `topic_id` 和 `chapter` 为准。

## 对应章节

{links}

## 示例统计

- 共 {len(directory_entries)} 个示例：{tiers}
- 每个示例保留 `# ---` metadata、`if __name__ == "__main__"` 入口和 `OK` 成功标记。

## 运行与验收

```powershell
python code/scripts/run_all_examples.py --chapter ch{int(directory.name[2:4]):02d} --tier core
python code/scripts/run_all_examples.py --chapter ch{int(directory.name[2:4]):02d} --tier llm
python code/scripts/run_all_examples.py --chapter ch{int(directory.name[2:4]):02d} --tier gpu
```

默认 LLM/GPU 验收使用 mock 或条件跳过；真实 API、GPU、模型下载和付费调用必须显式启用。
"""
            (directory / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    return manifest


def verify_metadata(manifest: dict) -> list[str]:
    failures: list[str] = []
    for entry in manifest["examples"]:
        path = CODE / entry["path"]
        text = path.read_text(encoding="utf-8")
        expected = (
            f"# chapter: {entry['chapter']}",
            f"# topic_id: {entry['topic_id']}",
            f"# See: ../../../{entry['chapter_file']}",
        )
        for marker in expected:
            if marker not in text:
                failures.append(f"{entry['path']}: missing {marker}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite metadata, READMEs and manifest")
    args = parser.parse_args()
    generated = build(write=args.write)
    failures = verify_metadata(generated)
    if MANIFEST.is_file() and not args.write:
        current = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if current != generated:
            failures.append("TOPIC_MANIFEST.json is stale; run with --write")
    elif not MANIFEST.is_file():
        failures.append("TOPIC_MANIFEST.json is missing; run with --write")
    for failure in failures[:30]:
        print(f"[FAIL] {failure}")
    if failures:
        return 1
    print(f"[PASS] topic manifest: {generated['example_count']} unique examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
