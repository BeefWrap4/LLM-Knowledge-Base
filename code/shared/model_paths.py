"""Resolve the tutorial's external model storage directory."""

from __future__ import annotations

import os
from pathlib import Path

WINDOWS_DEFAULT_MODELS_DIR = Path(
    r"E:\AI_Models\Projects\MyDocument\Python到大模型应用_面试教程_2026版\models"
)
POSIX_DEFAULT_MODELS_DIR = Path.home() / ".cache" / "llm-interview-tutorial" / "models"


def tutorial_models_dir() -> Path:
    """Return the external model directory, honoring ``TUTORIAL_MODELS_DIR``."""
    configured = os.environ.get("TUTORIAL_MODELS_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        return WINDOWS_DEFAULT_MODELS_DIR
    return POSIX_DEFAULT_MODELS_DIR
