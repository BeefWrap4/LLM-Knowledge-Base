# ---
# chapter: 16
# topic: ORPO Loss (真实可微 PyTorch 实现)
# section: 16.11.5
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 15_orpo_loss.py
# expected_runtime: <3s
# expected_output: ORPO loss 值 + .backward() 验证可微
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.11.5
#
# Interview hooks:
#   1. ORPO 相对 DPO 的优势？单模型训练省一半显存？SFT + DPO 一步完成？
#   2. Odds Ratio 相比 log-ratio 的工程意义？更稳定还是更敏感？
#   3. 五大 DAA 算法选择决策：什么场景选 ORPO？什么场景选 KTO？
"""ORPO (Odds Ratio Preference Optimization) loss 训练 demo.

ORPO = SFT loss + odds ratio loss (无需 reference model, 比 DPO 更简单):
  L = L_sft + λ * L_OR
  L_OR = -log σ(β * (log_odds(θ_chosen) - log_odds(θ_rejected)))

  其中 log_odds(θ) = log P(y|x) - log(1 - P(y|x))

ORPO 优势:
  - 单模型训练, 显存 ~ 省 50% (无需 ref policy)
  - SFT + 对齐一步完成
  - 在 7B 量级上比 DPO 更快收敛
"""
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch
import torch.nn.functional as F
from shared.gpu_guard import require_nvidia_gpu
from shared._error_helper import raise_with_help


def check_hardware():
    require_nvidia_gpu(min_vram_gb=0, min_count=1)


def log_odds(logp: torch.Tensor) -> torch.Tensor:
    """log p / (1 - p) (加 epsilon 防 log(0))."""
    return logp - torch.log1p(-logp.exp() + 1e-8)


def orpo_loss(
    chosen_logps: torch.Tensor,    # [B, T] chosen 序列 token-level logp
    rejected_logps: torch.Tensor,  # [B, T] rejected 序列 token-level logp
    chosen_labels: torch.Tensor,   # [B, T] label mask (-100 表示 padding/prompt)
    beta: float = 0.1,
    lam: float = 0.5,
) -> torch.Tensor:
    """ORPO 总损失 = SFT loss + λ * odds ratio loss."""
    # 1) SFT loss: chosen 上 NLL (mask 掉 prompt 部分)
    mask = (chosen_labels != -100).float()
    sft = -(chosen_logps * mask).sum() / mask.sum().clamp_min(1.0)

    # 2) Odds ratio: 序列级 log_odds
    chosen_seq_logp = (chosen_logps * mask).sum(dim=-1)
    rejected_seq_logp = (rejected_logps * mask).sum(dim=-1)
    log_odds_ratio = log_odds(chosen_seq_logp) - log_odds(rejected_seq_logp)

    # 3) 偏好损失: 鼓励 log_odds_ratio > 0
    pref = -F.logsigmoid(beta * log_odds_ratio).mean()

    return sft + lam * pref


def main():
    check_hardware()

    print("=== ORPO Loss (真实 PyTorch) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    torch.manual_seed(0)
    B, T = 4, 32
    chosen_logps = (torch.randn(B, T, device="cuda") * 0.1 - 2.0).requires_grad_(True)
    rejected_logps = chosen_logps.detach() - torch.rand(B, T, device="cuda") * 0.5
    chosen_labels = torch.randint(0, 1000, (B, T), device="cuda")
    chosen_labels[:, :5] = -100  # mask 掉 prompt 部分

    loss = orpo_loss(chosen_logps, rejected_logps, chosen_labels)
    loss.backward()

    print(f"  chosen_logps shape:    {tuple(chosen_logps.shape)}")
    print(f"  rejected_logps shape:  {tuple(rejected_logps.shape)}")
    print(f"  loss:                 {loss.item():.4f}")
    print(f"  loss requires_grad:   {chosen_logps.requires_grad}")
    print(f"  chosen.grad norm:     {chosen_logps.grad.norm().item():.4f}")
    print("\n  ORPO loss 可微, 可直接用于偏好对齐训练")
    print("  优势: 单模型 (无需 ref), SFT+对齐一步完成, 显存 ~ 省 50%")


if __name__ == "__main__":
    main()
