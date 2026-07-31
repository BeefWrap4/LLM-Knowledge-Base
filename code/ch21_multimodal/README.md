# Ch21 — 多模态大模型

> 教程: [`../tutorial/21_多模态大模型.md`](../tutorial/21_多模态大模型.md)

## 例子

| Tier | Files | 主题 |
|------|-------|------|
| gpu | 11 | CLIP / LLaVA / Diffusion LLM / Cosmos 3 条件接口 / 实时语音 |

## 快速开始

```bash
cd code/
python scripts/run_all_examples.py --tier gpu --chapter ch21  # 离线：结构示例或明确 SKIP

# 真实 OpenCLIP：会使用 GPU/权重/本地图像
CH21_OPENCLIP_RUN=1 CH21_IMAGE=/path/to/image.jpg \
  python ch21_multimodal/gpu/01_openclip_zero_shot.py
```

`04/06/07/10/11` 是明确标注的结构或算法骨架，不代表加载了真实 VLM、执行了训练、
连接了 Moshi，或完成了端到端多模态 RAG。

## 关联章节

- Ch12: Transformer 跨模态复用
- Ch26: 世界模型专题
- Ch28: 多模态在端侧
