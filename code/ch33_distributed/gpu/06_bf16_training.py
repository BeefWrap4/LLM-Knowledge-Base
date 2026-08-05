# ---
# chapter: 33
# topic: 大模型分布式训练
# topic_id: distributed.bf16_training
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 06_bf16_training.py
# expected_runtime: <5s
# expected_output: BF16 autocast training logs
# ---
# See: ../../../33_大模型分布式训练.md
#
# Interview hooks:
# 1. BF16 的 8 位指数位带来什么根本优势?
# 2. 训练 7B+ 模型时, BF16 与 FP16 在硬件支持上有何不同?
# 3. PyTorch 2.x 中 torch.autocast 与 torch.cuda.amp.autocast 的关系?


"""
支持 BF16 的训练硬件上，BF16 通常比 FP16 更易保持数值范围；仍需按硬件和模型实测。
"""
try:
    import torch
except ImportError:
    print("[SKIP] 需要 torch；请安装 GPU tier 依赖")
    print("OK")
    raise SystemExit(0)
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
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=2)


# ============================================================
# 方式一: PyTorch 原生 BF16 (推荐, A100/H100 首选)
# ============================================================
def train_with_bf16():
    """小型 BF16 autocast 训练 smoke。"""
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
    "bf16": {"enabled": True},
    "fp16": {
        "enabled": False  # 只启用 BF16
    },
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
