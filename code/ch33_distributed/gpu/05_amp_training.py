# ---
# chapter: 33
# topic: 大模型分布式训练
# topic_id: distributed.amp_training
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 05_amp_training.py
# expected_runtime: <5s (concept demo)
# expected_output: AMP loss-scaling logic walkthrough
# ---
# See: ../../../33_大模型分布式训练.md
#
# Interview hooks:
# 1. FP16 的最小可表示正数是多少? 为什么会产生梯度下溢?
# 2. GradScaler 在检测到 inf/nan 时如何处理? scale 怎么变化?
# 3. 为什么 BF16 训练不需要 GradScaler?
# PyTorch 的自动混合精度示例 (AMP)


try:
    import torch
except ImportError:
    print("[SKIP] 需要 torch；请安装 GPU tier 依赖")
    print("OK")
    raise SystemExit(0)


# ============================================================
# Mock 模型 + dataloader 用于演示
# ============================================================
class MyModel(torch.nn.Module):
    """简单的两层 MLP, 用于演示 AMP 的工作流程"""

    def __init__(self, in_dim=64, hidden=128, out_dim=10):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_dataloader(n=8, in_dim=64):
    """构造一个 8 个样本的 mock dataloader"""
    x = torch.randn(n, in_dim)
    y = torch.randint(0, 10, (n,))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=2)


# ============================================================
# 核心: train_with_amp
# ============================================================
def train_with_amp(model, dataloader, optimizer):
    """
    使用 FP16 AMP 进行混合精度训练
    注意: BF16 不需要 GradScaler!
    """
    from torch.amp import GradScaler, autocast

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    # FP16 loss scaling 只在 CUDA 路径启用；CPU smoke 保持 FP32。
    scaler = GradScaler("cuda", init_scale=2**16, enabled=use_cuda)
    print(f"[AMP] device={device}, initial scale={scaler.get_scale()}")

    for step, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        # autocast 自动将 CUDA 前向传播转为 FP16
        with autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            outputs = model(x)
            loss = torch.nn.functional.cross_entropy(outputs, y)

        # 反向传播: scaler 自动处理梯度缩放
        # scaler.scale(loss) 将 loss 乘以 scale_factor
        # backward 后 scaler 会自动检测溢出 (inf/nan)
        scaler.scale(loss).backward()

        # 如果有溢出, 跳过本次更新并缩小 scale
        # 如果没有溢出, 正常更新并尝试增大 scale
        scaler.step(optimizer)
        scaler.update()

        print(f"  Step {step}, loss={loss.item():.4f}, scale={scaler.get_scale():.0f}")
        if step >= 2:
            break

    # 内部逻辑 (简化):
    # if 检测到 INF/NaN:
    #     skip update, scale *= 0.5 (backoff_factor)
    # else:
    #     unscale gradients, optimizer.step()
    #     scale *= 2.0 (growth_factor)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = make_dataloader()
    train_with_amp(model, dataloader, optimizer)


if __name__ == "__main__":
    main()
    print("OK")
