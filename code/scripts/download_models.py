#!/usr/bin/env python3
# ---
# code/scripts/download_models.py
# 从国内源 (ModelScope / hf-mirror) 下载教程所需模型
# Usage: python code/scripts/download_models.py [--list | --required-only | ...]
# ---
"""
下载模型到仓库外的 ``TUTORIAL_MODELS_DIR``。默认模型使用 ModelScope →
Hugging Face 镜像 → Hugging Face 官方端点的顺序；需要精确文件过滤的模型直接使用
Hugging Face。

注册模型（截至 2026-07-31；默认只下载 * 标记的 3 个）:
  * bge-small-zh-v1.5        (embedding)       [默认下载]
  * bge-reranker-v2-m3       (reranker)        [默认下载]
  * Qwen2.5-0.5B-Instruct    (llm-small)       [默认下载]
  - Qwen2.5-7B-Instruct      (llm-medium)      [--llm-medium --confirm-large]
  - Llama-3.1-8B-Instruct    (llm-medium)      [--llm-medium --confirm-large, 需授权]
  - Cosmos3-Nano             (world-model)     [--world-model --confirm-large]
  - lerobot/pi0_base         (vla)             [--world-model --confirm-large]
  - DeepSeek-R1-Distill-1.5B (reasoner)        [--reasoner]
  - Qwen2.5-7B-4bit-mlx      (edge-mlx)        [--edge-mlx, Apple Silicon]
  - Llama-3.2-3B Q4_K_M      (edge-gguf)       [--edge-gguf，仅下载指定 GGUF]
  - Qwen2.5-0.5B-lora        (training)        [--training, depends on qwen0_5b]
  - Qwen2.5-0.5B-ddp         (training)        [--training, depends on qwen0_5b]

下载源策略:
  1. ModelScope（若仓库存在）
  2. Hugging Face 镜像 hf-mirror.com
  3. Hugging Face 官方端点

``size_gb`` 仅是规划值，不是承诺；仓库 revision、dtype、量化和筛选文件会改变实际下载量。
大型/受许可模型必须先阅读当前模型卡，并显式传入 ``--confirm-large``。
"""

import argparse
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from shared.model_paths import tutorial_models_dir  # noqa: E402

MODELS = tutorial_models_dir()


MODELS_TO_DOWNLOAD = {
    # === Embedding / Rerank (默认下载) ===
    "bge-small-zh": {
        "model_id": "BAAI/bge-small-zh-v1.5",
        "local_name": "bge-small-zh-v1.5",
        "size_gb": 0.1,
        "tier": "embedding",
        "chapters": ["ch14_rag", "ch17_evaluation", "ch20_llmops", "ch22_data_eng"],
        "required": True,
    },
    "bge-reranker": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "local_name": "bge-reranker-v2-m3",
        "size_gb": 0.6,
        "tier": "reranker",
        "chapters": ["ch17_evaluation", "ch22_data_eng"],
        "required": True,
    },
    # === LLM 小模型 (默认下载) ===
    "qwen0_5b": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_name": "Qwen2.5-0.5B-Instruct",
        "size_gb": 1.0,
        "tier": "llm-small",
        "chapters": [
            "ch12_transformer_architecture",
            "ch13_prompt_engineering",
            "ch14_rag",
            "ch15_agent",
            "ch16_finetuning",
            "ch17_evaluation",
            "ch18_llm_frameworks",
            "ch19_distributed",
            "ch29_context_engineering",
        ],
        "required": True,
    },
    # === LLM 中等 (--llm-medium 时下载) ===
    "qwen7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_name": "Qwen2.5-7B-Instruct",
        "size_gb": 15.0,
        "tier": "llm-medium",
        "chapters": ["ch25_inference_engines"],
        "required": False,
        "requires_confirmation": True,
    },
    "llama8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_name": "Llama-3.1-8B-Instruct",
        "size_gb": 16.0,
        "tier": "llm-medium",
        "chapters": ["ch25_inference_engines"],
        "required": False,
        "needs_auth": True,
        "requires_confirmation": True,
    },
    "cosmos3-nano": {
        "model_id": "nvidia/Cosmos3-Nano",
        "local_name": "Cosmos3-Nano",
        # 多组件仓库的实际体积必须按当前 revision 和所选组件 dry-run。
        "size_gb": None,
        "tier": "world-model",
        "chapters": ["ch26_world_models"],
        "required": False,
        "requires_confirmation": True,
    },
    "pi0-vla": {
        "model_id": "lerobot/pi0_base",
        "local_name": "pi0_base",
        "size_gb": 14.0,
        "tier": "vla",
        "chapters": ["ch26_world_models"],
        "required": False,
        "requires_confirmation": True,
    },
    "r1-distill-1_5b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "local_name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "size_gb": 3.0,
        "tier": "reasoner",
        "chapters": ["ch27_reasoning_ttc"],
        "required": False,
    },
    "mlx-qwen7b-4bit": {
        "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "local_name": "Qwen2.5-7B-Instruct-4bit-mlx",
        "size_gb": 5.0,
        "tier": "edge-mlx",
        "chapters": ["ch28_edge_llm"],
        "required": False,
        "platform": "apple-silicon",
    },
    "llama-cpp-3b": {
        "model_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "local_name": "Llama-3.2-3B-Instruct-GGUF",
        "size_gb": 2.0,
        "tier": "edge-gguf",
        "chapters": ["ch28_edge_llm"],
        "required": False,
        "allow_patterns": ["Llama-3.2-3B-Instruct-Q4_K_M.gguf", "README.md"],
        "sentinel": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "hf_only": True,
    },
    # === 训练辅助 (复用 qwen0_5b) ===
    "qwen0_5b-lora": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_name": "Qwen2.5-0.5B-Instruct-lora",
        "size_gb": 0.1,  # 仅 LoRA adapters
        "tier": "training",
        "chapters": ["ch16_finetuning"],
        "required": False,
        "depends_on": "qwen0_5b",
    },
    "qwen0_5b-ddp": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_name": "Qwen2.5-0.5B-Instruct-ddp",
        "size_gb": 1.0,
        "tier": "training",
        "chapters": ["ch19_distributed"],
        "required": False,
        "depends_on": "qwen0_5b",
    },
}


def download_modelscope(repo_id: str, local_dir: Path) -> bool:
    """通过 ModelScope SDK 下载."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("  [WARN] modelscope 未安装, pip install modelscope")
        return False
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
        return True
    except Exception as e:
        print(f"  [WARN] modelscope 下载失败: {e}")
        return False


def download_hf_snapshot(
    repo_id: str,
    local_dir: Path,
    *,
    endpoint: str | None,
    allow_patterns: list[str] | None = None,
) -> bool:
    """通过指定 Hugging Face endpoint 下载，可精确限制文件集合。"""
    label = endpoint or "https://huggingface.co"
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            endpoint=endpoint,
            allow_patterns=allow_patterns,
            max_workers=4,
        )
        return True
    except ImportError:
        print("  [WARN] huggingface_hub 未安装, pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  [WARN] Hugging Face endpoint {label} 下载失败: {e}")
        return False


def download(
    repo_id: str,
    local_name: str,
    *,
    allow_patterns: list[str] | None = None,
    sentinel: str = "config.json",
    hf_only: bool = False,
) -> bool:
    local_dir = MODELS / local_name
    if (local_dir / sentinel).exists():
        print(f"  [SKIP] {local_name} 已存在 ({local_dir})")
        return True
    print(f"\n  下载 {repo_id} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # ModelScope 的文件过滤语义与 HF 不同；精确筛选模型不走此分支。
    if not hf_only and allow_patterns is None and download_modelscope(repo_id, local_dir):
        return True
    if download_hf_snapshot(
        repo_id,
        local_dir,
        endpoint="https://hf-mirror.com",
        allow_patterns=allow_patterns,
    ):
        return True
    if download_hf_snapshot(
        repo_id,
        local_dir,
        endpoint=None,
        allow_patterns=allow_patterns,
    ):
        return True
    print(f"  [FAIL] 所有源都失败, 请手动下载 {repo_id}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--list", action="store_true", help="只列出注册模型与门禁，不下载")
    grp.add_argument("--all", action="store_true", help="选择全部注册模型（大型模型仍需 --confirm-large）")
    grp.add_argument("--embedding", action="store_true", help="仅 embedding (bge-small-zh)")
    grp.add_argument("--reranker", action="store_true", help="仅 reranker (bge-reranker-v2-m3)")
    grp.add_argument("--llm", action="store_true", help="仅小型 LLM (qwen0_5b)")
    grp.add_argument("--llm-medium", action="store_true", help="中等 LLM (qwen7b + llama8b)")
    grp.add_argument("--world-model", action="store_true", help="Physical AI 模型 (cosmos3-nano + pi0-vla)")
    grp.add_argument("--reasoner", action="store_true", help="推理模型 (r1-distill-1_5b)")
    grp.add_argument("--edge-mlx", action="store_true", help="Apple Silicon 量化 (mlx-qwen7b-4bit)")
    grp.add_argument("--edge-gguf", action="store_true", help="GGUF 量化 (llama-cpp-3b)")
    grp.add_argument("--training", action="store_true", help="训练辅助 (qwen0_5b-lora + qwen0_5b-ddp)")
    grp.add_argument("--required-only", action="store_true", help="仅下载 required=True 的模型 (默认 3 个)")
    ap.add_argument(
        "--confirm-large",
        action="store_true",
        help="确认已阅读模型卡/许可证并核对磁盘、显存与网络预算",
    )
    args = ap.parse_args()

    if args.list:
        print("key\ttier\trequired\tconfirmation\tplanning_size_gb\tmodel_id")
        for key, info in MODELS_TO_DOWNLOAD.items():
            size = info.get("size_gb")
            print(
                f"{key}\t{info['tier']}\t{bool(info.get('required'))}\t"
                f"{bool(info.get('requires_confirmation'))}\t"
                f"{size if size is not None else 'unknown'}\t{info['model_id']}"
            )
        print("\nsize_gb 仅用于初步规划；下载前以当前仓库 revision/dry-run 为准。")
        return 0

    # 根据 tier / required 选 targets
    targets = []
    if args.all:
        targets = list(MODELS_TO_DOWNLOAD.keys())
    elif args.required_only:
        targets = [k for k, v in MODELS_TO_DOWNLOAD.items() if v.get("required")]
    elif args.embedding:
        targets = ["bge-small-zh"]
    elif args.reranker:
        targets = ["bge-reranker"]
    elif args.llm:
        targets = ["qwen0_5b"]
    elif args.llm_medium:
        targets = [k for k, v in MODELS_TO_DOWNLOAD.items() if v["tier"] == "llm-medium"]
    elif args.world_model:
        targets = [k for k, v in MODELS_TO_DOWNLOAD.items() if v["tier"] in ("world-model", "vla")]
    elif args.reasoner:
        targets = [k for k, v in MODELS_TO_DOWNLOAD.items() if v["tier"] == "reasoner"]
    elif args.edge_mlx:
        targets = [k for k, v in MODELS_TO_DOWNLOAD.items() if v["tier"] == "edge-mlx"]
    elif args.edge_gguf:
        targets = [k for k, v in MODELS_TO_DOWNLOAD.items() if v["tier"] == "edge-gguf"]
    elif args.training:
        targets = [k for k, v in MODELS_TO_DOWNLOAD.items() if v["tier"] == "training"]
    else:
        # 默认仅选择 required=True；实际体积以当前 revision/dry-run 为准。
        targets = [k for k, v in MODELS_TO_DOWNLOAD.items() if v.get("required")]

    unconfirmed = [key for key in targets if MODELS_TO_DOWNLOAD[key].get("requires_confirmation")]
    if unconfirmed and not args.confirm_large:
        print(f"[REFUSE] 大型或受许可模型需要显式 --confirm-large: {unconfirmed}")
        print("请先阅读当前模型卡/许可证，并用 Hub dry-run 核对实际下载量。")
        return 2

    print("=" * 60)
    print("  模型下载器 (国内源优先)")
    print("=" * 60)
    print(f"  目标: {targets}")
    print(f"  目录: {MODELS}")
    print()

    MODELS.mkdir(parents=True, exist_ok=True)
    failed = []
    for key in targets:
        info = MODELS_TO_DOWNLOAD[key]
        if not download(
            info["model_id"],
            info["local_name"],
            allow_patterns=info.get("allow_patterns"),
            sentinel=info.get("sentinel", "config.json"),
            hf_only=info.get("hf_only", False),
        ):
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
