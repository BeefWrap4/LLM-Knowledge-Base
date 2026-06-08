#!/usr/bin/env python3
# ---
# code/scripts/download_models.py
# 从国内源 (ModelScope / hf-mirror) 下载教程所需模型
# Usage: python code/scripts/download_models.py [--all | --embedding | --reranker | --llm]
# ---
"""
下载模型到 code/models/ 目录, 国内源优先 (ModelScope).

支持的模型 (12 个, 默认下载 * 标记的):
  * bge-small-zh-v1.5       (0.1GB, embedding)     [默认下载]
  * bge-reranker-v2-m3      (0.6GB, reranker)      [默认下载]
  * Qwen2.5-0.5B-Instruct   (1.0GB, llm-small)     [默认下载]
  - Qwen2.5-7B-Instruct     (15GB,  llm-medium)    [--llm-medium]
  - Llama-3.1-8B-Instruct   (16GB,  llm-medium)    [--llm-medium, needs_auth]
  - Cosmos-1.0-7B           (14GB,  world-model)   [--world-model]
  - Pi0-VLA-base            (8GB,   vla)           [--vla]
  - DeepSeek-R1-Distill-1.5B (3GB,  reasoner)      [--reasoner]
  - Qwen2.5-7B-4bit-mlx     (5GB,   edge-mlx)      [--edge-mlx, Apple Silicon]
  - llama-3.2-3b-q4_k_m.gguf (2GB,  edge-gguf)     [--edge-gguf]
  - Qwen2.5-0.5B-lora       (0.1GB, training)      [--training, depends on qwen0_5b]
  - Qwen2.5-0.5B-ddp        (1.0GB, training)      [--training, depends on qwen0_5b]

下载源策略:
  1. ModelScope (国内 CDN, 5-10 MB/s)
  2. HuggingFace 镜像 hf-mirror.com (1-3 MB/s)
  3. HuggingFace 直连 (海外用户)
"""

import argparse
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
MODELS = CODE / "models"


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
    },
    "llama8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_name": "Llama-3.1-8B-Instruct",
        "size_gb": 16.0,
        "tier": "llm-medium",
        "chapters": ["ch25_inference_engines"],
        "required": False,
        "needs_auth": True,
    },
    "cosmos7b": {
        "model_id": "nvidia/Cosmos-1.0-7B",
        "local_name": "Cosmos-1.0-7B",
        "size_gb": 14.0,
        "tier": "world-model",
        "chapters": ["ch26_world_models"],
        "required": False,
    },
    "pi0-vla": {
        "model_id": "lerobot/pi0-base",
        "local_name": "Pi0-VLA-base",
        "size_gb": 8.0,
        "tier": "vla",
        "chapters": ["ch26_world_models"],
        "required": False,
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
        "model_id": "TheBloke/Llama-3.2-3B-Instruct-GGUF",
        "local_name": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.0,
        "tier": "edge-gguf",
        "chapters": ["ch28_edge_llm"],
        "required": False,
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


def download_hf_mirror(repo_id: str, local_dir: Path) -> bool:
    """通过 HF 镜像 (hf-mirror.com) 下载."""
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        from huggingface_hub import snapshot_download

        # max_workers=4 并行下载, 比默认 8 更稳 (CDN 限流时不会拖慢)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            env=env,
            max_workers=4,
        )
        return True
    except ImportError:
        print("  [WARN] huggingface_hub 未安装, pip install huggingface_hub")
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
    grp.add_argument("--all", action="store_true", help="下载全部 12 个模型 (含 LLM, 边端等)")
    grp.add_argument("--embedding", action="store_true", help="仅 embedding (bge-small-zh)")
    grp.add_argument("--reranker", action="store_true", help="仅 reranker (bge-reranker-v2-m3)")
    grp.add_argument("--llm", action="store_true", help="仅小型 LLM (qwen0_5b)")
    grp.add_argument("--llm-medium", action="store_true", help="中等 LLM (qwen7b + llama8b)")
    grp.add_argument("--world-model", action="store_true", help="世界模型 (cosmos7b + pi0-vla)")
    grp.add_argument("--reasoner", action="store_true", help="推理模型 (r1-distill-1_5b)")
    grp.add_argument("--edge-mlx", action="store_true", help="Apple Silicon 量化 (mlx-qwen7b-4bit)")
    grp.add_argument("--edge-gguf", action="store_true", help="GGUF 量化 (llama-cpp-3b)")
    grp.add_argument("--training", action="store_true", help="训练辅助 (qwen0_5b-lora + qwen0_5b-ddp)")
    grp.add_argument("--required-only", action="store_true", help="仅下载 required=True 的模型 (默认 3 个)")
    args = ap.parse_args()

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
        # 默认: required=True 的 (~1.7GB)
        targets = [k for k, v in MODELS_TO_DOWNLOAD.items() if v.get("required")]

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
        if not download(info["model_id"], info["local_name"]):
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
