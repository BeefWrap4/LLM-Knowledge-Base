# ---
# chapter: 16
# topic: ORPO Loss (真实可微 PyTorch 实现)
# section: 16.10.5
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 15_orpo_loss.py
# expected_runtime: <3s
# expected_output: ORPO loss 值 + .backward() 验证可微
# ---
# See: ../../../16_模型微调与推理优化.md §16.10.5
#
# Interview hooks:
#   1. ORPO 的 reference-free 目标为什么不等于固定节省 50% 显存？
#   2. ORPO 如何把 chosen NLL 与 odds-ratio 偏好项联合优化？
#   3. 如何用同一基座/数据预算比较 ORPO、DPO 与其他偏好方法？
"""ORPO (Odds Ratio Preference Optimization) loss 训练 demo.

ORPO = chosen NLL + odds-ratio preference loss (无需 reference model):
  L = L_sft + λ * L_OR
  L_OR = -log σ(log_odds(θ_chosen) - log_odds(θ_rejected))

  其中 log_odds(θ) = log P(y|x) - log(1 - P(y|x))

ORPO 优势:
  - 单模型训练, 无需常驻 reference policy
  - chosen NLL 与偏好目标在一个训练阶段联合优化
  - 实际显存与收敛收益取决于实现、序列长度、优化器和分片策略
"""

import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import torch
import torch.nn.functional as F

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    require_nvidia_gpu(min_vram_gb=0, min_count=1)


def log_odds(logp: torch.Tensor) -> torch.Tensor:
    """序列概率的 log odds；限制上界以避免 log(0)."""
    probability = logp.exp().clamp(max=1.0 - 1e-7)
    return logp - torch.log1p(-probability)


def orpo_loss(
    chosen_logps: torch.Tensor,  # [B, T] chosen 序列 token-level logp
    rejected_logps: torch.Tensor,  # [B, T] rejected 序列 token-level logp
    chosen_labels: torch.Tensor,  # [B, T] label mask (-100 表示 padding/prompt)
    rejected_labels: torch.Tensor,  # [B, T] rejected 自己的 mask
    lam: float = 0.1,
) -> torch.Tensor:
    """ORPO 总损失 = SFT loss + λ * odds ratio loss."""
    # 1) SFT loss: chosen 上 NLL (mask 掉 prompt 部分)
    chosen_mask = (chosen_labels != -100).float()
    rejected_mask = (rejected_labels != -100).float()
    sft = -(chosen_logps * chosen_mask).sum() / chosen_mask.sum().clamp_min(1.0)

    # 2) Odds ratio: 序列级 log_odds
    chosen_seq_logp = (chosen_logps * chosen_mask).sum(dim=-1)
    rejected_seq_logp = (rejected_logps * rejected_mask).sum(dim=-1)
    log_odds_ratio = log_odds(chosen_seq_logp) - log_odds(rejected_seq_logp)

    # 3) 偏好损失: 鼓励 log_odds_ratio > 0
    pref = -F.logsigmoid(log_odds_ratio).mean()

    return sft + lam * pref


def main():
    check_hardware()

    print("=== ORPO Loss (真实 PyTorch) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    torch.manual_seed(0)
    B, T = 4, 32
    chosen_logps = (torch.randn(B, T, device="cuda") * 0.1 - 2.0).requires_grad_(True)
    rejected_logps = (torch.randn(B, T, device="cuda") * 0.1 - 2.3).requires_grad_(True)
    chosen_labels = torch.randint(0, 1000, (B, T), device="cuda")
    chosen_labels[:, :5] = -100  # mask 掉 prompt 部分
    rejected_labels = chosen_labels.clone()
    rejected_labels[:, -2:] = -100  # rejected 可有不同 completion 长度

    loss = orpo_loss(chosen_logps, rejected_logps, chosen_labels, rejected_labels)
    loss.backward()

    print(f"  chosen_logps shape:    {tuple(chosen_logps.shape)}")
    print(f"  rejected_logps shape:  {tuple(rejected_logps.shape)}")
    print(f"  loss:                 {loss.item():.4f}")
    print(f"  loss requires_grad:   {chosen_logps.requires_grad}")
    print(f"  chosen.grad norm:     {chosen_logps.grad.norm().item():.4f}")
    print(f"  rejected.grad norm:   {rejected_logps.grad.norm().item():.4f}")
    print("\n  教学 ORPO loss 可微")
    print("  结构: 无需 reference policy，chosen NLL 与偏好目标联合优化")
    print("  注意: 不能据此固定宣称节省 50% 显存，需按训练栈实测")
    print("OK")


if __name__ == "__main__":
    main()
