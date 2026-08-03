# ---
# code/tests/test_download_models.py
# 测试 download_models 注册表的安全默认值与当前模型标识
# ---
"""测试 download_models 注册表结构、默认范围与精确文件筛选。"""

import sys
from pathlib import Path

# 把 scripts/ 加入 sys.path (scripts/ 没有 __init__.py)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import download_models  # noqa: E402
from download_models import MODELS_TO_DOWNLOAD  # noqa: E402


def test_required_defaults_are_small_explicit_set():
    """注册新模型不能静默扩大默认下载集合。"""
    required = {key for key, info in MODELS_TO_DOWNLOAD.items() if info.get("required")}
    assert required == {"bge-small-zh", "bge-reranker", "qwen0_5b"}


def test_models_dir_honors_environment_override(monkeypatch, tmp_path):
    """模型目录可迁移到仓库外，并由单一环境变量覆盖。"""
    target = tmp_path / "external-models"
    monkeypatch.setenv("TUTORIAL_MODELS_DIR", str(target))

    assert download_models.tutorial_models_dir() == target


def test_models_dict_keys():
    """每个模型必须有可定位的仓库、目录与 tier。"""
    required = ["model_id", "local_name", "tier"]
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
        "cosmos3-nano",
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


def test_models_dict_size_gb_is_optional_planning_value():
    """未知的多组件仓库可不写体积；若填写则必须为正数。"""
    for key, info in MODELS_TO_DOWNLOAD.items():
        size_gb = info.get("size_gb")
        assert size_gb is None or size_gb > 0, f"{key} size_gb={size_gb} 非正数"


def test_current_physical_ai_model_ids():
    """不得重新引入退役 Cosmos-1 标识或错误的 Pi0 连字符仓库名。"""
    all_ids = {info["model_id"] for info in MODELS_TO_DOWNLOAD.values()}
    assert "nvidia/Cosmos3-Nano" in all_ids
    assert "lerobot/pi0_base" in all_ids
    assert not any("Cosmos-1" in model_id for model_id in all_ids)
    assert "lerobot/pi0-base" not in all_ids


def test_gguf_download_is_filtered_to_one_quant():
    """GGUF 仓库包含多种量化，下载器必须限制到教程所需文件。"""
    info = MODELS_TO_DOWNLOAD["llama-cpp-3b"]
    assert info["model_id"] == "bartowski/Llama-3.2-3B-Instruct-GGUF"
    assert info["sentinel"] == "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    assert info["sentinel"] in info["allow_patterns"]
    assert info["hf_only"] is True


def test_list_mode_is_read_only(monkeypatch, tmp_path, capsys):
    """--list 不能创建模型目录，更不能进入下载函数。"""
    models_dir = tmp_path / "models"
    monkeypatch.setattr(download_models, "MODELS", models_dir)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--list"])
    monkeypatch.setattr(
        download_models,
        "download",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    assert download_models.main() == 0
    output = capsys.readouterr().out
    assert "nvidia/Cosmos3-Nano" in output
    assert "confirmation" in output
    assert not models_dir.exists()
