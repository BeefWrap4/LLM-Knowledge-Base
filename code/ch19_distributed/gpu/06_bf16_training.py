# ---
# chapter: 19
# topic: 分布式训练系统 - BF16 混合精度训练
# section: 19.7.3
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 06_bf16_training.py
# expected_runtime: <5s
# expected_output: BF16 autocast training logs
# ---
# See: ../tutorial/19_分布式训练系统.md#1973-automatic-mixed-precision-amp
#
# Interview hooks:
# 1. BF16 的 8 位指数位带来什么根本优势?
# 2. 训练 7B+ 模型时, BF16 与 FP16 在硬件支持上有何不同?
# 3. PyTorch 2.x 中 torch.autocast 与 torch.cuda.amp.autocast 的关系?


# === Multi-GPU / heavy model guard (auto-added) ===
import sys as _sys
import os as _os
_NGPU = _os.environ.get("WORLD_SIZE", "1")
if _NGPU == "1" and not _os.environ.get("FORCE_GPU_RUN"):
    print(f"[SKIP] {{__file__}}: 需多卡 (WORLD_SIZE>1) 或真实模型权重, 用 torchrun 或设置 FORCE_GPU_RUN=1")
    _sys.exit(0)
"""
混合精度训练的最佳实践 —— BF16 优先 (2026 年推荐)
"""
import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
# Mock 模型 + dataloader
# ============================================================
class MyModel(nn.Module):
    def __init__(self, in_dim=64, hidden=128, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_data(n=8, in_dim=64):
    x = torch.randn(n, in_dim)
    y = torch.randint(0, 10, (n,))
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=2
    )


# ============================================================
# 方式一: PyTorch 原生 BF16 (推荐, A100/H100 首选)
# ============================================================
@torch.no_grad()
def train_with_bf16():
    """最简单的 BF16 训练 —— 零代码侵入"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = make_data()

    print(f"[BF16] device={device}")
    for step, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        # 只需一行: 开启 BF16 autocast
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)

        # 反向传播自动用 BF16 (梯度可能自动提升到 FP32)
        loss.backward()
        optimizer.step()
        # 注意: BF16 不需要 GradScaler!
        print(f"  Step {step}, loss={loss.item():.4f}")
        if step >= 2:
            break


# ============================================================
# 方式二: DeepSpeed 的 BF16 配置 (JSON 模板, 这里转成 Python dict)
# ============================================================
DEEPSPEED_BF16_CONFIG = {
    "bf16": {
        "enabled": True
    },
    "fp16": {
        "enabled": False   # 只启用 BF16
    }
}


# ============================================================
# 方式三: FSDP + BF16
# ============================================================
def make_fsdp_bf16_policy():
    """FSDP 的 BF16 MixedPrecision 策略"""
    try:
        from torch.distributed.fsdp import MixedPrecision
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    except ImportError:
        return None


def main():
    print("=" * 60)
    print("BF16 vs FP16 关键差异:")
    print("=" * 60)
    print("  FP16: 5 位指数 + 10 位尾数, 范围小, 需要 Loss Scaling")
    print("  BF16: 8 位指数 +  7 位尾数, 范围与 FP32 相同, 无需 Loss Scaling")
    print("=" * 60)

    train_with_bf16()
    print(f"DeepSpeed BF16 config: {DEEPSPEED_BF16_CONFIG}")
    mp = make_fsdp_bf16_policy()
    if mp is not None:
        print(f"FSDP MixedPrecision policy created: {mp}")


if __name__ == "__main__":
    main()
    print("OK")