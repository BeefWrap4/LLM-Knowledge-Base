"""统一错误格式化 — 所有 RuntimeError 走这个出口."""
from typing import Optional


def format_error(
    message: str,
    hint: str,
    file_path: Optional[str] = None,
    line: Optional[int] = None,
) -> str:
    """生成统一格式的错误信息.

    格式:
        [ERROR] {file}:{line}  {message}
        [HELP]  {hint}
        [HELP]  或 `export LLM_MOCK=1` 用 mock 跑 (仅 CI/离线)
    """
    location = ""
    if file_path:
        location = f"{file_path}"
        if line is not None:
            location += f":{line}"
        location = f"{location}  "
    parts = [f"[ERROR] {location}{message}"]
    parts.append(f"[HELP]  {hint}")
    parts.append("[HELP]  或 `export LLM_MOCK=1` 用 mock 跑 (仅 CI/离线)")
    return "\n".join(parts)


def raise_with_help(message: str, hint: str, exc_class=RuntimeError) -> None:
    """抛带 help 信息的异常."""
    raise exc_class(format_error(message, hint))
