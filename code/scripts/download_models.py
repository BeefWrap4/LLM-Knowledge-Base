#!/usr/bin/env python3
# ---
# code/scripts/download_models.py
# 从国内源 (ModelScope / hf-mirror) 下载教程所需模型
# Usage: python code/scripts/download_models.py [--all | --embedding | --reranker | --llm]
# ---
"""
下载模型到 code/models/ 目录, 国内源优先 (ModelScope).

支持的模型:
  - bge-small-zh-v1.5    (100MB, embedding)        [默认下载]
  - bge-reranker-v2-m3   (600MB, reranker)         [默认下载]
  - Qwen2.5-0.5B-Instruct (1GB, 小型 LLM)          [--all 时下载]

下载源策略:
  1. ModelScope (国内 CDN, 5-10 MB/s)
  2. HuggingFace 镜像 hf-mirror.com (1-3 MB/s)
  3. HuggingFace 直连 (海外用户)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
MODELS = CODE / "models"


MODELS_TO_DOWNLOAD = {
    "embedding": ("BAAI/bge-small-zh-v1.5", "bge-small-zh-v1.5"),
    "reranker":  ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"),
    "llm":       ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
}


def download_modelscope(repo_id: str, local_dir: Path) -> bool:
    """通过 ModelScope SDK 下载."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print(f"  [WARN] modelscope 未安装, pip install modelscope")
        return False
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
        return True
    except Exception as e:
        print(f"  [WARN] modelscope 下载失败: {e}")
        return False


def download_hf_mirror(repo_id: str, local_dir: Path) -> bool:
    """通过 HF 镜像 (hf-mirror.com) 下载."""
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir), env=env)
        return True
    except ImportError:
        print(f"  [WARN] huggingface_hub 未安装, pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  [WARN] hf-mirror 下载失败: {e}")
        return False


def download(repo_id: str, local_name: str) -> bool:
    local_dir = MODELS / local_name
    if (local_dir / "config.json").exists():
        print(f"  [SKIP] {local_name} 已存在 ({local_dir})")
        return True
    print(f"\n  下载 {repo_id} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # 策略 1: ModelScope (国内优先)
    if download_modelscope(repo_id, local_dir):
        return True
    # 策略 2: HF 镜像
    if download_hf_mirror(repo_id, local_dir):
        return True
    # 策略 3: 失败
    print(f"  [FAIL] 所有源都失败, 请手动下载 {repo_id}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--all", action="store_true", help="下载全部 (含 LLM)")
    grp.add_argument("--embedding", action="store_true", help="仅 embedding")
    grp.add_argument("--reranker", action="store_true", help="仅 reranker")
    grp.add_argument("--llm", action="store_true", help="仅小 LLM")
    args = ap.parse_args()

    # 默认: embedding + reranker (~700MB)
    targets = []
    if args.all:
        targets = list(MODELS_TO_DOWNLOAD.keys())
    elif args.embedding:
        targets = ["embedding"]
    elif args.reranker:
        targets = ["reranker"]
    elif args.llm:
        targets = ["llm"]
    else:
        targets = ["embedding", "reranker"]

    print("=" * 60)
    print("  模型下载器 (国内源优先)")
    print("=" * 60)
    print(f"  目标: {targets}")
    print(f"  目录: {MODELS}")
    print()

    MODELS.mkdir(parents=True, exist_ok=True)
    failed = []
    for key in targets:
        repo_id, local_name = MODELS_TO_DOWNLOAD[key]
        if not download(repo_id, local_name):
            failed.append(key)

    print("\n" + "=" * 60)
    if failed:
        print(f"  ✗ 部分失败: {failed}")
        print("=" * 60)
        return 1
    print(f"  ✓ 全部下载完成, 共 {len(targets)} 个")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
