# 本地模型目录

此目录用于存放按需下载的模型权重，不应提交到 Git。

在 `code/` 目录先查看只读清单：

```bash
python scripts/download_models.py --list
```

常用入口：

```bash
python scripts/download_models.py --required-only
python scripts/download_models.py --embedding
python scripts/download_models.py --reranker
python scripts/download_models.py --llm
python scripts/download_models.py --llm-medium --confirm-large
python scripts/download_models.py --world-model --confirm-large
```

大型或受许可模型必须显式传 `--confirm-large`。这只表示你确认了下载意图，不替代对当前模型卡、
许可、磁盘、显存、后端与用途限制的检查。

下载器会优先尝试 ModelScope，再尝试 Hugging Face 镜像和官方端点；实际速度和体积不作承诺。
GGUF 条目会用文件过滤避免下载同仓库的全部量化版本。

完整说明见 [`../docs/MODELS.md`](../docs/MODELS.md)。
