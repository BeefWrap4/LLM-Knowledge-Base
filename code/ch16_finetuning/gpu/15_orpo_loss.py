# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.11.5 ORPO 损失核心 (Direct Alignment Algorithms)
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 15_orpo_loss.py
# expected_runtime: <3s
# expected_output: 在 mock 数据上计算 ORPO 损失的 4 个分量
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.11.5
# Interview hooks:
#   1. ORPO 相对 DPO 的优势？单模型训练省一半显存？SFT + DPO 一步完成？
#   2. Odds Ratio 相比 log-ratio 的工程意义？更稳定还是更敏感？
#   3. 五大 DAA 算法选择决策：什么场景选 ORPO？什么场景选 KTO？

"""
ORPO 损失核心 (PyTorch 极简实现)

ORPO = Odds Ratio Preference Optimization
- 单模型训练（无需参考模型）, 节省 ~50% 显存
- 一步完成 SFT + DPO 风格对齐
- 目标: L_SFT - lambda * log_sigmoid(beta * log_odds_ratio)
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def log_odds(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """
    log p(y|x) - log(1-p(y|x)) 的近似实现
    工程上常用 log p / log(1-p) 比例代替
    """
    odd = logp - torch.log(1.0 - logp.exp() + 1e-8)
    odd_ref = logp_ref - torch.log(1.0 - logp_ref.exp() + 1e-8)
    return odd - odd_ref


def orpo_loss(
    policy_chosen: torch.Tensor,     # (T_w,)  chosen 序列的对数概率和
    policy_rejected: torch.Tensor,   # (T_l,)  rejected 序列的对数概率和
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
    lam: float = 0.5,
):
    """
    ORPO 总损失 = SFT 损失 + lambda * (-偏好损失)
    """
    # 1) SFT 损失 (仅在 chosen 上, 类 cross-entropy)
    sft = -policy_chosen.mean()

    # 2) Odds Ratio: log odds(chosen) - log odds(rejected)
    log_odds_ratio = (
        log_odds(policy_chosen, ref_chosen)
        - log_odds(policy_rejected, ref_rejected)
    )

    # 3) 偏好损失: 鼓励 log_odds_ratio > 0
    pref = F.logsigmoid(beta * log_odds_ratio).mean()

    # 4) 总损失
    total = sft + lam * (-pref)
    return total, sft.detach(), pref.detach()


if __name__ == "__main__":
    torch.manual_seed(0)

    # 模拟 4 对偏好数据
    B, T = 4, 32
    policy_chosen = torch.randn(B, T) * 0.3 - 1.0     # 负值 (低概率)
    policy_rejected = torch.randn(B, T) * 0.3 - 2.0   # 更低
    ref_chosen = torch.randn(B, T) * 0.3 - 1.2
    ref_rejected = torch.randn(B, T) * 0.3 - 2.0

    # 聚合为序列级 log prob (求和)
    pc = policy_chosen.sum(-1)
    pr = policy_rejected.sum(-1)
    rc = ref_chosen.sum(-1)
    rr = ref_rejected.sum(-1)

    total, sft, pref = orpo_loss(pc, pr, rc, rr, beta=0.1, lam=0.5)
    print(f"[ORPO mock] sft_loss       = {sft.item():.4f}")
    print(f"[ORPO mock] preference     = {pref.item():.4f}  (越高越好)")
    print(f"[ORPO mock] total_loss     = {total.item():.4f}  (lam=0.5, beta=0.1)")
    print()
    print("ORPO 优势：单模型, 无需参考策略, SFT + 对齐一步完成, 显存 ~ 省 50%")
    print()
