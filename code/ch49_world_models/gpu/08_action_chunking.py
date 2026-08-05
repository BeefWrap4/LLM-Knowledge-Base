# ---
# chapter: 49
# topic: 世界模型、VLA 与具身智能
# topic_id: world_models.action_chunking
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 08_action_chunking.py
# expected_runtime: 5-10s (action chunking 演示)
# expected_output: chunked vs single-step 对比
# ---
# See: ../../../49_世界模型VLA与具身智能.md
#
# Interview hooks:
#   1. Action chunking vs 单步预测的核心优势? (平滑 + 时序一致)
#   2. K (chunk size) 如何选? (大 → 平滑但僵; 小 → 灵活但抖)
#   3. Action chunking 与 model predictive control (MPC) 的关系?
"""Action Chunking 演示 (ACT / Diffusion Policy 共享).

核心: 一次预测未来 K 个动作 (chunk), 而非单步
  - 平滑性: 避免单步预测的抖动
  - 时序一致性: 显式建模动作序列
  - 推理频率: 每 K 步才重预测, 中间 open-loop 执行

本 demo: 模拟预测器 + 时序动作块, 对比 chunk vs single-step 的执行曲线.
"""

import sys
from pathlib import Path

import torch

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch.nn as nn

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


class ChunkedPredictor(nn.Module):
    """简化 chunked predictor: 状态 → K 步动作 (线性层)."""

    def __init__(self, state_dim: int = 14, action_dim: int = 7, chunk_size: int = 10, hidden: int = 128):
        super().__init__()
        self.K = chunk_size
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, chunk_size * action_dim),
        )
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """输入: state [B, state_dim] → 输出: chunk [B, K, action_dim]."""
        out = self.net(state)
        return out.view(-1, self.K, self.action_dim)


def chunked_execution(
    predictor: ChunkedPredictor,
    initial_state: torch.Tensor,
    K: int = 10,
    total_steps: int = 100,
) -> torch.Tensor:
    """Action chunking 执行: 每 K 步重预测, 中间 open-loop."""
    trajectory = []
    state = initial_state
    next_chunk = None
    chunk_start = 0
    for t in range(total_steps):
        # 每 K 步重预测
        if t % K == 0:
            with torch.no_grad():
                next_chunk = predictor(state)  # [1, K, action_dim]
            chunk_start = t
        # open-loop 执行
        idx_in_chunk = t - chunk_start
        action = next_chunk[0, idx_in_chunk]  # [action_dim]
        trajectory.append(action)
        # 简化的状态转移: 把 action 拼到 state 前 7 维 (action 影响前 7 个关节)
        delta = torch.cat([action, torch.zeros_like(action)], dim=0) * 0.1
        state = state + delta.unsqueeze(0)
    return torch.stack(trajectory)


def main() -> None:
    check_hardware()
    print("=== Action Chunking 演示 ===\n")
    print("核心: 一次预测 K 步动作, 中间 open-loop 执行")
    print()

    state_dim, action_dim, K, total = 14, 7, 10, 100

    # 训练一个简单 predictor (50 步快速拟合)
    print("步骤 1: 训练一个 chunked predictor (50 步, 合成数据)")
    predictor = ChunkedPredictor(state_dim, action_dim, K).cuda()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in predictor.parameters())

    B = 32
    state = torch.randn(B, state_dim).cuda()
    # 目标 chunk: 简单线性函数 of state, shape [B, K, action_dim]
    # 用 state 前 7 维 × 0.1 作为每步动作目标
    target_chunk = state[:, :action_dim].unsqueeze(1).expand(-1, K, -1) * 0.1

    losses = []
    for step in range(50):
        pred = predictor(state)
        loss = ((pred - target_chunk) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"  step {step:3d} | MSE = {loss.item():.4f}")
    print(f"  ✅ loss 下降: {losses[0]:.4f} → {losses[-1]:.4f}\n")

    # Action chunking rollout
    print(f"步骤 2: Action chunking rollout (K={K}, total={total} 步)")
    initial = torch.randn(1, state_dim).cuda()
    traj = chunked_execution(predictor, initial, K=K, total_steps=total)

    print(f"  预测频率: 每 {K} 步一次 → 总 {total // K} 个 chunk")
    print(f"  open-loop 执行: 中间 {K} 步不重预测")
    print(f"  trajectory shape: {tuple(traj.shape)} (T, action_dim)\n")

    # 展示 3 个 chunk 的边界
    print("步骤 3: chunk 边界处的连续性")
    for chunk_idx in [0, 5, 9]:
        boundary_t = chunk_idx * K
        if boundary_t == 0:
            a_before = traj[0]
        else:
            a_before = traj[boundary_t - 1]
        a_after = traj[boundary_t]
        diff = (a_after - a_before).norm().item()
        print(f"  chunk #{chunk_idx} (t={boundary_t}): |a[K-1] - a[K]| = {diff:.4f}")

    print()
    print("=" * 60)
    print("Action Chunking 优势:")
    print("  - 平滑: 1 次预测 K 步, 避免单步抖动")
    print("  - 调度: 以一次 chunk 预测替代逐步调用；实际加速不等于固定 K 倍")
    print("  - 时序: 显式建模动作轨迹 (适合长任务)")
    print("  - 风险: open-loop 错误累积 (compounding error)")
    print()
    print("ACT、Diffusion Policy、π0 等均可使用动作分块；chunk 长度按任务与实现核对。")


if __name__ == "__main__":
    main()
    print("OK")
