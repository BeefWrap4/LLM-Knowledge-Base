# ---
# chapter: 33
# topic: 大模型分布式训练
# topic_id: distributed.muon_optimizer
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 10_muon_optimizer.py
# expected_runtime: <5s (无 torch.compile 时)
# expected_output: Muon optimizer 状态演示
# ---
# See: ../../../33_大模型分布式训练.md
#
# Interview hooks:
# 1. Muon 优化器的"零次幂"指什么? 为什么 5 次 Newton-Schulz 迭代就够了?
# 2. Muon 的显存开销比 AdamW 少多少? 为什么?
# 3. 为什么 Muon 只用于 >= 2D 参数 (矩阵), 而 embedding 仍用 AdamW?
# 4. Muon 与 ZeRO-3 兼容吗? 它的动量如何被分片?


import os as _os

try:
    import torch
except ImportError:
    print("[SKIP] 需要 torch；请安装 GPU tier 依赖")
    print("OK")
    raise SystemExit(0)
from torch import Tensor

# 编译会产生较大一次性开销；默认 smoke 使用普通函数，显式开启才编译。
if _os.environ.get("CH19_MUON_COMPILE") == "1" and hasattr(torch, "compile"):
    _compile = torch.compile
else:

    def _compile(fn, **_kw):
        return fn


@_compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz 迭代近似计算 G 的"零次幂" (即正交化)。
    实测 5 次迭代即可得到足够精确的 U。
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT  # 始终让较小维度在最后, 便于迭代
    # 归一化: 谱范数 <= 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon 核心步骤的教学实现，不替代 DeepSpeed/PyTorch 生产实现。
    用法: optimizer = Muon(model_params, lr=0.02, momentum=0.95)
    生产实践通常只将符合条件的隐藏层矩阵交给 Muon，其余参数交给 AdamW。
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # 1) 动量 (类似 SGD with momentum)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf
                # 2) Decoupled weight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                # 3) 矩阵参数走 Newton-Schulz 正交化
                if g.ndim == 2:
                    g = zeropower_via_newtonschulz5(g)
                # 4) 应用更新 (统一学习率, 不做 per-param scale)
                p.data.add_(g, alpha=-lr)
        return None


def main():
    """用一个 2 层的 toy 模型演示 Muon 优化器的 step"""
    torch.manual_seed(0)
    # 矩阵参数 (走 Newton-Schulz)
    mat = torch.nn.Linear(64, 64, bias=False)
    # 1D 参数 (走普通动量, 跳过正交化)
    vec = torch.nn.Linear(64, 1, bias=True)

    muon = Muon(
        [
            {"params": mat.parameters(), "lr": 0.02, "momentum": 0.95},
            {"params": vec.parameters(), "lr": 0.02, "momentum": 0.95},
        ]
    )

    print("=" * 60)
    print("Muon 优化器演示")
    print("=" * 60)
    print(f"参数组数: {len(muon.param_groups)}")
    for i, g in enumerate(muon.param_groups):
        n_params = sum(p.numel() for p in g["params"])
        print(f"  组 {i}: lr={g['lr']}, 动量={g['momentum']}, 参数量={n_params}")

    # 模拟一次 step
    x = torch.randn(4, 64)
    target = torch.randn(4, 1)
    y = mat(x)
    y = vec(y)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    print(f"Loss before step: {loss.item():.4f}")
    muon.step()
    muon.zero_grad()
    print("=" * 60)
    print("DeepSpeed 配置入口（当前官方支持 ZeRO 1/2/3）:")
    print('  "optimizer": {"type": "Muon", "params": {"lr": 0.001, "momentum": 0.95}}')
    print("  参数分组、ns_method 与 ZeRO/offload 选项请以当前 DeepSpeed 文档为准")
    print("=" * 60)


if __name__ == "__main__":
    main()
    print("OK")
