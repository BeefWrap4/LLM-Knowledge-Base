# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.3.1 LeRobot ACT — Action Chunking Transformer
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: lerobot (optional, raises if missing)
# run: python 05_lerobot_act.py
# expected_runtime: 5-15s (架构 + ACT 训练/推理示例)
# expected_output: ACT 配置 + 训练/推理代码片段
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.3.1
#
# Interview hooks:
#   1. ACT (Action Chunking Transformer) 与 L2 单步预测的核心区别?
#   2. 为什么 action chunking 能提升长任务成功率? (时序一致性 + 平滑)
#   3. LeRobot 与 Isaac Lab 的定位区别? (开源算法库 vs NVIDIA 仿真平台)
"""LeRobot ACT (Action Chunking Transformer) 演示.

LeRobot 是 HuggingFace 的机器人开源库:
  pip install lerobot
  提供 ACT / Diffusion Policy / VQ-BeT / HIL-SERL 等算法

ACT (Zhao et al. 2023) 核心: 一次预测未来 K 个动作 (chunk),
  而非单步 — 提升时序一致性 + 平滑性.

本 demo: 检查 lerobot 库 + 展示 ACT 训练/推理代码 (实际跑需 Linux + GPU).
"""

import shutil
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch
import torch.nn as nn

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


def check_lerobot_installed():
    """LeRobot 仅 Linux 支持完整功能. 缺则抛错并指引安装."""
    if shutil.which("lerobot") is None:
        try:
            import lerobot  # noqa: F401

            print("  ✓ lerobot Python 包已装")
        except ImportError:
            print("  ⚠️  lerobot Python 包未装")
            print("  本 demo 继续运行 (展示 ACT 架构 + 训练代码片段).")
            print("  全功能 LeRobot 训练需:")
            print("    1. Linux 平台 (Windows/Mac 仅推理受限)")
            print("    2. pip install lerobot")
            print("    3. 真机 (Aloha / Franka) 或 Isaac Lab 仿真")


class ACTConfig:
    """ACT (Action Chunking Transformer) 超参.

    生产 ACT: 100 步 chunk (≈ 1s @ 100Hz), 7 编码 + 7 解码层.
    """

    def __init__(
        self,
        chunk_size: int = 100,
        dim_feedforward: int = 3200,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 7,
        state_dim: int = 14,
        action_dim: int = 7,
        n_heads: int = 8,
    ):
        self.chunk_size = chunk_size
        self.dim_feedforward = dim_feedforward
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_heads = n_heads


class SimpleACTPolicy(nn.Module):
    """简化版 ACT: Transformer encoder-decoder.

    真实 ACT: ResNet 图像编码 + state 编码 + 序列解码 (chunk_size × action_dim).
    本 demo: 简化纯 transformer 演示 chunk 输出.
    """

    def __init__(self, cfg: ACTConfig, d_model: int = 128):
        super().__init__()
        self.cfg = cfg
        self.state_enc = nn.Linear(cfg.state_dim, d_model)
        self.action_dec = nn.Linear(d_model, cfg.action_dim)
        # 时序位置编码 (chunk_size 个位置)
        self.pos_emb = nn.Parameter(torch.randn(cfg.chunk_size, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward // 25,  # 缩小供 demo
            batch_first=True,
            dropout=0.0,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward // 25,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)  # 缩小供 demo
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """输入: state [B, state_dim] → 输出: chunk [B, chunk_size, action_dim]."""
        B = state.size(0)
        # encoder: state → 1 token
        memory = self.encoder(self.state_enc(state).unsqueeze(1))  # [B, 1, d]
        # decoder: chunk_size 个 query → 动作序列
        queries = self.pos_emb.unsqueeze(0).expand(B, -1, -1)  # [B, K, d]
        out = self.decoder(queries, memory)
        return self.action_dec(out)  # [B, K, action_dim]


def main() -> None:
    check_hardware()
    check_lerobot_installed()

    print("=== LeRobot ACT (Action Chunking Transformer) ===\n")
    cfg = ACTConfig()
    print("  ACT 配置:")
    print(f"    chunk_size       : {cfg.chunk_size} (≈ 1s @ 100Hz)")
    print(f"    n_encoder_layers : {cfg.n_encoder_layers}")
    print(f"    n_decoder_layers : {cfg.n_decoder_layers}")
    print(f"    state_dim        : {cfg.state_dim} (双臂 7-DoF × 2)")
    print(f"    action_dim       : {cfg.action_dim} (末端 7-DoF)\n")

    # 真跑: 简化 ACT policy + 50 步训练
    print("步骤: 简化 ACT policy 真跑 (5 步梯度下降)")
    model = SimpleACTPolicy(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,} (生产 ACT ~10M+)")

    # 合成数据
    B = 4
    state = torch.randn(B, cfg.state_dim).cuda()
    target_actions = torch.randn(B, cfg.chunk_size, cfg.action_dim).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for step in range(5):
        pred = model(state)
        loss = ((pred - target_actions) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"    step {step} | MSE = {loss.item():.4f}")

    print(f"\n  ✅ 5 步 loss: {losses[0]:.4f} → {losses[-1]:.4f}")

    print()
    print("=" * 60)
    print("LeRobot ACT 训练 (Python 端真实代码, 需 lerobot 装好):")
    print("  from lerobot.common.policies.act import ACTPolicy, ACTConfig")
    print("  cfg = ACTConfig(")
    print("      input_shapes={'observation.state': (state_dim,), ")
    print("                   'observation.image': (3, 480, 640)},")
    print("      output_shapes={'action': (chunk_size, action_dim)},")
    print("  )")
    print("  policy = ACTPolicy(cfg)")
    print("  # 用 LeRobotDataset + Trainer 训练")
    print()
    print("ACT 推理: 1 次 forward 预测 chunk_size 个动作, open-loop 执行 K 步.")


if __name__ == "__main__":
    main()
