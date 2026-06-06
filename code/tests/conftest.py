# ---
# code/tests/conftest.py
# pytest fixtures + path setup for code companion
# ---
"""
See: code/README.md §验证
"""
import sys
from pathlib import Path

# 把 code/ 加入 sys.path, 这样 "from shared import ..." 可工作
CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import pytest


@pytest.fixture
def code_root() -> Path:
    """Return the code/ root path."""
    return CODE_ROOT


@pytest.fixture
def chapter_dir(code_root):
    """Factory: 返回某章的路径."""
    def _make(chapter: str) -> Path:
        return code_root / chapter
    return _make
