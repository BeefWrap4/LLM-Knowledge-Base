#!/usr/bin/env python3
# ---
# code/scripts/verify_all.py
# 一键验证: 教程 wiki 链接 + 章节 README 覆盖 + core/ 例子跑通
# Usage: python code/scripts/verify_all.py
# ---
"""
Master verification script. Returns 0 iff all checks pass.

Checks (10 项):
  1. Wiki link integrity: 所有 [[WikiLinks]] 都能解析
  2. Repository consistency: 40 章、433 示例、编号唯一、示例契约
  3. Chapter README coverage: 29/29 代码章节都有 README.md
  4. Code companion health:
     - 每章都有 core/ 或 llm/ 或 gpu/
     - 每章 .py 数 >= 1
  5. Tutorial ↔ Code reference integrity + coverage baseline
  6. Source ledger: 40 章均有权威来源入口
  7. Documentation snapshot consistency
  8. CI LLM_MOCK safety
  9. Smoke test sample: 跑 5 个代表性 core 例子
 10. LLM doctor: 仅 --real-api --confirm-real 与 LLM_MOCK=0 同时启用
"""

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
REPO = CODE.parent
EXPECTED_CHAPTERS = 40
EXPECTED_CODE_CHAPTERS = 29
EXPECTED_EXAMPLES = 433
NUMBERED_HEADING_RE = re.compile(r"^#{2,6}\s+(\d{1,2}(?:\.\d+)+)\b")
PYTHON_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_./\\-])"
    r"((?:code[\\/])?ch\d{2}_[A-Za-z0-9_.-]+[\\/]"
    r"(?:core|llm|gpu)[\\/][A-Za-z0-9_.-]+\.py)"
)
MERMAID_START_RE = re.compile(r"^\s*```mermaid\s*$")
MARKDOWN_FENCE_END_RE = re.compile(r"^\s*```\s*$")
MERMAID_ALLOWED_DIAGRAMS = {
    "architecture-beta",
    "block-beta",
    "C4Component",
    "C4Container",
    "C4Context",
    "C4Deployment",
    "C4Dynamic",
    "classDiagram",
    "classDiagram-v2",
    "erDiagram",
    "flowchart",
    "gantt",
    "gitGraph",
    "graph",
    "journey",
    "kanban",
    "mindmap",
    "packet-beta",
    "pie",
    "quadrantChart",
    "requirementDiagram",
    "sankey-beta",
    "sequenceDiagram",
    "stateDiagram",
    "stateDiagram-v2",
    "timeline",
    "xychart-beta",
    "zenuml",
}
MERMAID_RISK_PATTERNS = (
    (
        "unquoted nested '[' in a square node label",
        re.compile(r'\b[A-Za-z_][A-Za-z0-9_-]*\[(?!["(])[^\]\r\n]*\['),
    ),
    (
        "unquoted '{' or '}' in a square node label",
        re.compile(r'\b[A-Za-z_][A-Za-z0-9_-]*\[(?!["(])[^\]\r\n]*[{}]'),
    ),
    (
        "unquoted '(' or ')' in a square node label",
        re.compile(r'\b[A-Za-z_][A-Za-z0-9_-]*\[(?!["(])[^\]\r\n]*[()]'),
    ),
    (
        "mismatched quoted square node label",
        re.compile(r'\b[A-Za-z_][A-Za-z0-9_-]*\["[^"\r\n]*"\)'),
    ),
)


def canonical_chapters() -> list[Path]:
    """Return Ch01-Ch40 canonical Markdown files, excluding index/report files."""
    return [
        path
        for path in sorted(REPO.glob("[0-9][0-9]_*.md"))
        if path.name not in {"00_目录索引.md", "99_库健康检查报告.md"}
    ]


def check_wiki_links() -> bool:
    print("\n--- [1/10] Wiki link integrity ---")
    r = subprocess.run(
        [sys.executable, str(CODE / "scripts" / "verify_xrefs.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    # Filter out the summary lines we want
    for line in r.stdout.splitlines():
        if line.startswith("===") or "BROKEN" in line or "Resolved:" in line or "Broken:" in line:
            print(f"  {line}")
    return r.returncode == 0


def check_repo_consistency() -> bool:
    print("\n--- [2/10] Repository consistency ---")
    chapters = canonical_chapters()
    chapter_numbers = [int(path.name[:2]) for path in chapters]
    expected_numbers = list(range(1, EXPECTED_CHAPTERS + 1))
    code_chapters = sorted(path for path in CODE.glob("ch[0-9][0-9]_*") if path.is_dir())
    examples = sorted(CODE.glob("ch[0-9][0-9]_*/*/*.py"))
    required_locks = [
        CODE / "requirements-core.ci.lock",
        CODE / "requirements-llm.ci.lock",
        CODE / "requirements-gpu.ci.lock",
    ]
    missing_locks = [path.name for path in required_locks if not path.is_file()]
    gpu_contract_failures = find_gpu_mock_contract_failures()
    dynamic_execution_failures = find_dynamic_execution_failures(examples)
    mermaid_blocks, mermaid_format_failures = inspect_mermaid_blocks()

    duplicate_headings: list[str] = []
    for chapter in chapters:
        seen: dict[str, int] = {}
        for line_no, line in enumerate(chapter.read_text(encoding="utf-8").splitlines(), 1):
            match = NUMBERED_HEADING_RE.match(line)
            if not match:
                continue
            number = match.group(1)
            if number in seen:
                duplicate_headings.append(
                    f"{chapter.name}:{line_no} duplicates {number} (first at {seen[number]})"
                )
            else:
                seen[number] = line_no

    missing_metadata = []
    missing_main = []
    for example in examples:
        text = example.read_text(encoding="utf-8")
        rel = str(example.relative_to(REPO))
        if "# ---" not in "\n".join(text.splitlines()[:20]):
            missing_metadata.append(rel)
        if 'if __name__ == "__main__"' not in text and "if __name__ == '__main__'" not in text:
            missing_main.append(rel)

    print(f"  Chapters:      {len(chapters)}/{EXPECTED_CHAPTERS}")
    print(f"  Code chapters: {len(code_chapters)}/{EXPECTED_CODE_CHAPTERS}")
    print(f"  Examples:      {len(examples)}/{EXPECTED_EXAMPLES}")
    print(f"  Duplicate numbered headings: {len(duplicate_headings)}")
    print(f"  Missing metadata/main guard: {len(missing_metadata)}/{len(missing_main)}")
    print(f"  Missing CI locks: {len(missing_locks)}")
    print(f"  GPU mock contract failures: {len(gpu_contract_failures)}")
    print(f"  Builtin eval/exec calls: {len(dynamic_execution_failures)}")
    print(f"  Mermaid blocks/format failures: {mermaid_blocks}/{len(mermaid_format_failures)}")
    for issue in (
        duplicate_headings
        + missing_metadata
        + missing_main
        + missing_locks
        + gpu_contract_failures
        + dynamic_execution_failures
        + mermaid_format_failures
    )[:20]:
        print(f"  [FAIL] {issue}")

    return all(
        [
            chapter_numbers == expected_numbers,
            len(code_chapters) == EXPECTED_CODE_CHAPTERS,
            len(examples) == EXPECTED_EXAMPLES,
            not duplicate_headings,
            not missing_metadata,
            not missing_main,
            not missing_locks,
            not gpu_contract_failures,
            not dynamic_execution_failures,
            not mermaid_format_failures,
        ]
    )


def inspect_mermaid_blocks() -> tuple[int, list[str]]:
    """Fail closed on Mermaid constructs known to break Obsidian's parser.

    This lightweight CI gate covers fence integrity, diagram declarations, malformed quoted
    square nodes, and parser-sensitive characters in unquoted square labels. Exact release
    acceptance still uses the Mermaid version bundled with the target Obsidian installation.
    """
    total = 0
    failures: list[str] = []
    for markdown in sorted(REPO.glob("*.md")):
        lines = markdown.read_text(encoding="utf-8").splitlines()
        inside = False
        start_line = 0
        body: list[tuple[int, str]] = []
        for line_no, line in enumerate(lines, 1):
            if not inside and MERMAID_START_RE.match(line):
                inside = True
                start_line = line_no
                body = []
                continue
            if not inside:
                continue
            if MARKDOWN_FENCE_END_RE.match(line):
                total += 1
                meaningful = [
                    value.strip()
                    for _, value in body
                    if value.strip() and not value.lstrip().startswith("%%")
                ]
                if not meaningful:
                    failures.append(f"{markdown.name}:{start_line} empty Mermaid block")
                else:
                    diagram = meaningful[0].split(maxsplit=1)[0]
                    if diagram not in MERMAID_ALLOWED_DIAGRAMS:
                        failures.append(
                            f"{markdown.name}:{start_line + 1} unsupported Mermaid diagram: {diagram}"
                        )
                for body_line_no, body_line in body:
                    for description, pattern in MERMAID_RISK_PATTERNS:
                        if pattern.search(body_line):
                            failures.append(
                                f"{markdown.name}:{body_line_no} {description}; quote the label"
                            )
                inside = False
                body = []
                continue
            body.append((line_no, line))
        if inside:
            failures.append(f"{markdown.name}:{start_line} unclosed Mermaid fence")
    return total, failures


def _dotted_call_name(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def find_dynamic_execution_failures(examples: list[Path] | None = None) -> list[str]:
    """Reject builtin ``eval``/``exec`` in reader-facing examples.

    Removing ``__builtins__`` is not a security boundary for adversarial Python
    object graphs. Tool inputs should use a narrow parser or an isolated sandbox.
    """

    failures: list[str] = []
    scripts = examples if examples is not None else sorted(CODE.glob("ch[0-9][0-9]_*/*/*.py"))
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted_call_name(node.func)
            if name in {"eval", "exec", "builtins.eval", "builtins.exec"}:
                relative = script.relative_to(CODE).as_posix()
                failures.append(f"{relative}:{node.lineno}: builtin {name}() is forbidden")
    return failures


def _mock_safe_risky_calls(source: str) -> list[str]:
    """Find operations disallowed in a GPU example marked as locally mock-safe."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg}"]

    exact_names = {
        "OpenAI",
        "Anthropic",
        "Client",
        "hf_hub_download",
        "snapshot_download",
        "start_http_server",
    }
    risky_suffixes = {
        ".from_pretrained",
        ".write_text",
        ".write_bytes",
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "requests.get",
        "requests.post",
        "requests.put",
        "requests.delete",
        "requests.request",
        "urllib.request.urlopen",
        "socket.create_connection",
        "torch.hub.load",
    }
    failures: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_call_name(node.func)
        if name in exact_names or any(name.endswith(suffix) for suffix in risky_suffixes):
            failures.append(f"line {node.lineno}: {name}")
    return failures


def find_gpu_mock_contract_failures() -> list[str]:
    """Require every GPU example to skip mock mode or declare audited local execution."""
    failures: list[str] = []
    for script in sorted(CODE.glob("ch[0-9][0-9]_*/gpu/*.py")):
        source = script.read_text(encoding="utf-8")
        relative = script.relative_to(CODE).as_posix()
        header = "\n".join(source.splitlines()[:20])
        if "skip_if_mock(" in source:
            continue
        if "# mock_safe: true" not in header:
            failures.append(f"{relative}: missing skip_if_mock() or '# mock_safe: true'")
            continue
        for risky_call in _mock_safe_risky_calls(source):
            failures.append(f"{relative}: mock_safe code contains {risky_call}")
    return failures


def check_readme_coverage() -> bool:
    print("\n--- [3/10] Chapter README coverage ---")
    expected = EXPECTED_CODE_CHAPTERS
    chapters = sorted(path for path in CODE.glob("ch[0-9][0-9]_*") if path.is_dir())
    missing = [directory.name for directory in chapters if not (directory / "README.md").is_file()]
    actual = len(chapters) - len(missing)
    print(f"  Chapter READMEs: {actual}/{expected}")
    if missing:
        print(f"  Missing: {missing}")
    return len(chapters) == expected and not missing


def check_code_health() -> bool:
    print("\n--- [4/10] Code companion health ---")
    chapters = sorted(path for path in CODE.glob("ch[0-9][0-9]_*") if path.is_dir())
    total_py = 0
    unhealthy = []
    for ch in chapters:
        py_files = list(ch.glob("*/[!_]*.py"))  # exclude __init__ if any
        n = len(py_files)
        total_py += n
        if n == 0:
            unhealthy.append(ch.name)
    print(f"  Chapters: {len(chapters)}")
    print(f"  Total .py: {total_py}")
    if unhealthy:
        print(f"  Unhealthy chapters (no .py): {unhealthy}")
        return False
    return total_py == EXPECTED_EXAMPLES


def check_sync_links() -> bool:
    print("\n--- [5/10] Tutorial ↔ Code reference integrity ---")
    r = subprocess.run(
        [sys.executable, str(CODE / "scripts" / "sync_links.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    # Extract the key summary lines
    for line in r.stdout.splitlines():
        if (
            "教程章节总数" in line
            or "Code 例子含" in line
            or "教程章节有 code 覆盖" in line
            or line.startswith("=== PASS")
            or line.startswith("=== FAIL")
        ):
            print(f"  {line.strip()}")
    broken_paths = find_broken_python_references()
    print(f"  Markdown Python paths: {'PASS' if not broken_paths else 'FAIL'} ({len(broken_paths)} broken)")
    for source, line_no, reference in broken_paths[:20]:
        print(f"  [FAIL] {source}:{line_no} -> {reference}")
    return r.returncode == 0 and not broken_paths


def find_broken_python_references() -> list[tuple[str, int, str]]:
    """Return executable ``chNN_*/{core,llm,gpu}/*.py`` references that do not exist.

    Historical implementation plans under ``docs/superpowers`` are snapshots rather than
    reader-facing runbooks, so they are deliberately outside this gate.
    """
    failures: list[tuple[str, int, str]] = []
    for markdown in sorted(REPO.rglob("*.md")):
        relative = markdown.relative_to(REPO)
        if relative.parts[:2] == ("docs", "superpowers"):
            continue
        for line_no, line in enumerate(markdown.read_text(encoding="utf-8").splitlines(), 1):
            for match in PYTHON_REFERENCE_RE.finditer(line):
                reference = match.group(1).replace("\\", "/")
                target = REPO / reference if reference.startswith("code/") else CODE / reference
                if not target.is_file():
                    failures.append((relative.as_posix(), line_no, reference))
    return failures


def check_source_ledger() -> bool:
    print("\n--- [6/10] Authoritative source ledger ---")
    ledger = REPO / "docs" / "AUTHORITATIVE_SOURCES.md"
    if not ledger.is_file():
        print("  [FAIL] docs/AUTHORITATIVE_SOURCES.md is missing")
        return False
    text = ledger.read_text(encoding="utf-8")
    missing = [f"Ch{number:02d}" for number in range(1, EXPECTED_CHAPTERS + 1) if f"| Ch{number:02d} |" not in text]
    rows_without_links = [
        line.split("|")[1].strip()
        for line in text.splitlines()
        if re.match(r"^\|\s*Ch\d{2}\s*\|", line) and "https://" not in line
    ]
    print(f"  Chapter source entries: {EXPECTED_CHAPTERS - len(missing)}/{EXPECTED_CHAPTERS}")
    if missing:
        print(f"  [FAIL] Missing: {', '.join(missing)}")
    if rows_without_links:
        print(f"  [FAIL] Entries without HTTPS source: {', '.join(rows_without_links)}")
    return not missing and not rows_without_links


def check_doc_snapshot() -> bool:
    print("\n--- [7/10] Documentation snapshot consistency ---")
    requirements = {
        "README.md": ("40 章节", "433 个可运行代码示例"),
        "code/README.md": ("29 章", "433"),
        "00_目录索引.md": ("40 章", "7 大板块"),
        "99_库健康检查报告.md": ("2026-07-31", "433"),
        "Python到大模型应用_面试教程_2026版_思维导图.xmind.md": ("40 章", "433"),
    }
    failures = []
    for rel, needles in requirements.items():
        path = REPO / rel
        if not path.is_file():
            failures.append(f"{rel}: missing")
            continue
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:80])
        missing = [needle for needle in needles if needle not in head]
        if missing:
            failures.append(f"{rel}: missing {missing}")
    for failure in failures:
        print(f"  [FAIL] {failure}")
    if not failures:
        print("  Current counts and dates agree across 5 entry documents")
    return not failures


def check_ci_llm_mock_safety() -> bool:
    """CI 安全检查: 防止 PR check 意外调真实 API.

    规则:
    - 在 GitHub Actions CI 环境, LLM_MOCK 必须精确为 1
    - 本地未设置 LLM_MOCK 时默认离线；只有精确为 0 才允许真实 API
    - 本地开发不阻塞；CI 缺少 LLM_MOCK=1 时直接失败
    """
    print("\n--- [8/10] CI LLM_MOCK safety check ---")
    in_ci = bool(os.environ.get("CI"))
    mock_set = os.environ.get("LLM_MOCK") == "1"

    if in_ci and not mock_set:
        print("  [WARN] CI 环境未设 LLM_MOCK=1, 可能意外调真实 API")
        print("         建议: GitHub Actions workflow 加 `env: LLM_MOCK: '1'`")
    elif mock_set:
        print("  [OK]   LLM_MOCK=1, 走 mock 路径 (CI 友好)")
    else:
        print("  [INFO] 本地非 CI 环境, LLM_MOCK 未设 (默认离线，不读取或使用 Key)")
    return not in_ci or mock_set


def check_smoke() -> bool:
    print("\n--- [9/10] Smoke test sample (5 core/ files) ---")
    sample = [
        "ch01_python_basics/core/22_list_dict_basics.py",
        "ch02_mutability/core/01_is_vs_equals.py",
        "ch03_oop/core/01_singleton.py",
        "ch06_memory_gc/core/01_pymalloc_object_size.py",
        "ch07_data_structures/core/01_linked_list.py",
    ]
    failures = []
    for rel in sample:
        script = CODE / rel
        if not script.is_file():
            print(f"  FAIL  {rel} (missing)")
            failures.append((rel, "missing"))
            continue
        r = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, cwd=str(CODE), timeout=30
        )
        ok = r.returncode == 0 and "OK" in r.stdout
        mark = "OK  " if ok else "FAIL"
        print(f"  {mark}  {rel}")
        if not ok:
            failures.append((rel, r.stderr[:100]))
    return len(failures) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-api",
        action="store_true",
        help="显式执行真实 API doctor；默认验收绝不访问外部 LLM",
    )
    parser.add_argument(
        "--confirm-real",
        action="store_true",
        help="确认本次 doctor 会产生外部请求并可能计费；必须与 --real-api 同时使用",
    )
    args = parser.parse_args()
    if args.real_api != args.confirm_real:
        parser.error("--real-api and --confirm-real must be supplied together")
    if args.real_api and os.environ.get("LLM_MOCK") != "0":
        parser.error("real API verification requires exact LLM_MOCK=0")

    print("=" * 60)
    print("  TUTORIAL+CODE COMPANION VERIFICATION")
    print("=" * 60)

    r1 = check_wiki_links()
    r2 = check_repo_consistency()
    r3 = check_readme_coverage()
    r4 = check_code_health()
    r5 = check_sync_links()
    r6 = check_source_ledger()
    r7 = check_doc_snapshot()
    r8 = check_ci_llm_mock_safety()
    r9 = check_smoke()
    r10 = check_llm_doctor(real_api=args.real_api, confirm_real=args.confirm_real)

    print("\n" + "=" * 60)
    print(f"  Wiki links:        {'PASS' if r1 else 'FAIL'}")
    print(f"  Repo consistency:  {'PASS' if r2 else 'FAIL'}")
    print(f"  README coverage:   {'PASS' if r3 else 'FAIL'}")
    print(f"  Code health:       {'PASS' if r4 else 'FAIL'}")
    print(f"  Reference sync:    {'PASS' if r5 else 'FAIL'}")
    print(f"  Source ledger:     {'PASS' if r6 else 'FAIL'}")
    print(f"  Doc snapshot:      {'PASS' if r7 else 'FAIL'}")
    print(f"  LLM_MOCK safety:   {'PASS' if r8 else 'FAIL'}")
    print(f"  Smoke sample:      {'PASS' if r9 else 'FAIL'}")
    doctor_status = "PASS" if args.real_api and r10 else ("SKIP" if not args.real_api else "FAIL")
    print(f"  LLM doctor:        {doctor_status}")
    print("=" * 60)
    return 0 if all([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10]) else 1


def check_llm_doctor(*, real_api: bool, confirm_real: bool = False) -> bool:
    """Run external provider checks only with explicit operator opt-in."""
    print("\n--- [10/10] LLM doctor (explicit real API only) ---")
    if not real_api:
        print("  SKIP (offline verification; real checks require separate double confirmation)")
        return True
    if not confirm_real or os.environ.get("LLM_MOCK") != "0":
        print("  FAIL (requires --real-api --confirm-real and exact LLM_MOCK=0)")
        return False

    sys.path.insert(0, str(CODE))  # 让 shared 可 import
    try:
        from shared.provider_registry import get_provider

        provider_name = os.environ.get("LLM_PROVIDER", "").strip()
        if not provider_name:
            print("  FAIL (LLM_PROVIDER must explicitly select one provider)")
            return False
        has_key = get_provider(provider_name).has_key()
    except Exception as e:
        print(f"  FAIL  provider_registry 不可用: {e}")
        return False
    if not has_key:
        print("  FAIL (no LLM API key in env; cannot satisfy --real-api)")
        return False
    r = subprocess.run(
        [
            sys.executable,
            str(CODE / "scripts" / "llm_doctor.py"),
            "--provider",
            provider_name,
            "--confirm-real",
        ],
        capture_output=True,
        text=True,
        cwd=str(CODE),
        timeout=120,
    )
    for line in r.stdout.splitlines():
        if "[✓]" in line or "[✗]" in line or "passed" in line or "Result:" in line:
            print(f"  {line.strip()}")
    return r.returncode == 0


if __name__ == "__main__":
    sys.exit(main())
