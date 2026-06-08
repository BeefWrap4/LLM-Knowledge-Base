# ---
# code/tests/test_download_models.py
# 测试 download_models 字典结构与最小模型数
# ---
"""测试 download_models 字典结构和最小内容."""

import sys
from pathlib import Path

# 把 scripts/ 加入 sys.path (scripts/ 没有 __init__.py)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from download_models import MODELS_TO_DOWNLOAD  # noqa: E402


def test_models_dict_has_minimum_12():
    """教程需要至少 12 个模型."""
    assert len(MODELS_TO_DOWNLOAD) >= 12


def test_models_dict_keys():
    """每个模型必须有 model_id + local_name + size_gb + tier."""
    required = ["model_id", "local_name", "size_gb", "tier"]
    for key, info in MODELS_TO_DOWNLOAD.items():
        assert isinstance(info, dict), f"{key} info 不是 dict, 是 {type(info).__name__}"
        for field in required:
            assert field in info, f"{key} 缺 {field}"


def test_models_dict_contains_known():
    """已知必须有的模型 key（用 key 字符串而非 model_id, 因 model_id 可能更新）."""
    required = [
        "bge-small-zh",
        "bge-reranker",
        "qwen0_5b",
        "qwen7b",
        "llama8b",
        "cosmos7b",
        "pi0-vla",
        "r1-distill-1_5b",
        "mlx-qwen7b-4bit",
        "llama-cpp-3b",
    ]
    for r in required:
        assert r in MODELS_TO_DOWNLOAD, f"缺 {r}"


def test_models_dict_no_duplicate_local_name():
    """不能有重复的 local_name（避免下载冲突）."""
    names = [info["local_name"] for info in MODELS_TO_DOWNLOAD.values()]
    assert len(names) == len(set(names)), f"重复 local_name: {names}"


def test_models_dict_size_gb_positive():
    """每个模型 size_gb 必须是正数."""
    for key, info in MODELS_TO_DOWNLOAD.items():
        assert info["size_gb"] > 0, f"{key} size_gb={info['size_gb']} 非正数"
