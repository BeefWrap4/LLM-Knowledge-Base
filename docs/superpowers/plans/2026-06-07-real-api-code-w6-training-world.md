# W6 训练 / 世界模型实现计划 — Real API Code

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 把 `ch19` (DDP/FSDP) / `ch16` (SFT/LoRA) / `ch26` (世界模型) / `ch27` (推理模型) 涉及 GPU 的 30 个文件改为真实训练 / 真实前向。

**前置依赖：** W1-W5 完成。

**目标硬件：** NVIDIA GPU 24GB+ (DDP/FSDP/SFT) + 80GB+ (世界模型 / VLA)

---

## 文件清单

### 重点修改

- `code/ch19_distributed/gpu/02_ddp_training.py`
- `code/ch19_distributed/gpu/03_fsdp_training.py`
- `code/ch16_finetuning/gpu/01-08_*.py` (~8 个 SFT/LoRA/RLHF)
- `code/ch26_world_models/gpu/01-10_*.py` (10 个)
- `code/ch27_reasoning_ttc/gpu/` (如需要新建)

---

## 任务 1：`ch19/02_ddp_training.py` — 真实 `accelerate launch`

- [ ] **步骤 1：读现状**

- [ ] **步骤 2：删除 mock dataset + fake model**

- [ ] **步骤 3：用 `Qwen2.5-0.5B + 合成 dataset` 真实训练 100 step**

```python
# 02_ddp_training.py
import os
import sys
import torch
from pathlib import Path
_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu

def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=2)

def main():
    check_hardware()
    model_path = "code/models/Qwen2.5-0.5B-Instruct"
    if not Path(model_path).exists():
        from shared._error_helper import raise_with_help
        raise_with_help(f"需要 {model_path}", "运行 `make download-models-default`.")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import Dataset, DataLoader
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model = model.cuda()
    
    # DDP 初始化
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
    
    # 合成 dataset
    class SyntheticDataset(Dataset):
        def __len__(self): return 100
        def __getitem__(self, i): 
            return {"input_ids": torch.randint(0, 1000, (32,)), "labels": torch.randint(0, 1000, (32,))}
    
    loader = DataLoader(SyntheticDataset(), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for step, batch in enumerate(loader):
        if step >= 100: break
        batch = {k: v.cuda() for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if dist.get_rank() == 0 and step % 10 == 0:
            print(f"[step {step}] loss={out.loss.item():.4f}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：跑（NVIDIA 24GB+ × 2）**

```bash
cd code
accelerate launch --num_processes 2 ch19_distributed/gpu/02_ddp_training.py
```

预期：100 step 训练完成，rank 0 打印 loss。

- [ ] **步骤 5：Commit**

---

## 任务 2：`ch19/03_fsdp_training.py` — 真实 FSDP

模式同任务 1，区别是 `from torch.distributed.fsdp import FullyShardedDataParallel`。

---

## 任务 3：`ch16/01-sft, 02-lora, ...` — 真实 `trl.SFTTrainer` 跑 10 step

每个文件用 `trl.SFTTrainer` 或 `peft.LoraConfig` 真实跑 10 step。

---

## 任务 4：`ch26/01-10` — 真实加载 Cosmos/Pi0 跑 1 step 前向

```python
# ch26/01_genie3_world_model.py
def main():
    check_hardware()  # require_nvidia_gpu(80)
    from transformers import AutoModel
    model = AutoModel.from_pretrained("code/models/Cosmos-1.0-7B", torch_dtype=torch.bfloat16)
    model = model.cuda()
    # 真实 1 步前向
    out = model(pixel_values=torch.randn(1, 3, 224, 224).cuda())
    print(f"Forward output shape: {out.last_hidden_state.shape}")

if __name__ == "__main__":
    main()
```

---

## 任务 5：教程 16, 19, 26, 27 章节同步更新

W7 wave 集中处理（不在 W6 范围）。

---

## 任务 6：Commit 收尾

```bash
git add -A
git commit -m "W6 training/world models: real DDP/FSDP/SFT/Cosmos forward passes"
```

---

## W6 验收清单

- [ ] `ch19/02` 真实 `accelerate launch` 跑 100 step
- [ ] `ch19/03` 真实 FSDP 跑 100 step
- [ ] `ch16/01-08` 真实 SFTTrainer / LoRA 跑 10 step
- [ ] `ch26/01-10` 真实加载 Cosmos/Pi0 跑 1 forward
- [ ] 缺 GPU / 缺权重时明确 RuntimeError
