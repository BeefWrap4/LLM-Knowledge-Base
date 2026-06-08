# ---
# chapter: 19
# topic: 分布式训练系统 - DataParallel 演示
# section: 19.2.1
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch
# run: python 01_dp_dataparallel.py
# expected_runtime: <5s (mock mode, no actual GPU required)
# expected_output: DP usage description
# ---
# See: ../tutorial/19_分布式训练系统.md#1921-dpdataparallel-原理与局限性
#
# Interview hooks:
# 1. 为什么 DP (DataParallel) 已被废弃? 说出至少 3 个致命缺陷。
# 2. DP 和 DDP 在进程模型、负载均衡、通信方式上的本质差异是什么?
# 3. 为什么 DP 不支持多机训练?


# === Multi-GPU / heavy model guard (auto-added) ===
import os as _os
import sys as _sys

_NGPU = _os.environ.get("WORLD_SIZE", "1")
if _NGPU == "1" and not _os.environ.get("FORCE_GPU_RUN"):
    print("[SKIP] {__file__}: 需多卡 (WORLD_SIZE>1) 或真实模型权重, 用 torchrun 或设置 FORCE_GPU_RUN=1")
    _sys.exit(0)
import torch
import torch.nn as nn


# ============================================================
# DataParallel 用法示例 (单进程多线程，不推荐)
# ============================================================
# 注意: 实际运行需要至少 1 张 CUDA GPU。
# 这里只演示 API 用法与原理。
# ============================================================
def demo_dp():
    """演示 nn.DataParallel 的最简用法"""
    if not torch.cuda.is_available():
        print("[Mock Mode] No CUDA device available, skipping actual run.")
        print("DP 工作流程:")
        print("  1. 将每个 batch 的数据均匀分配到所有 GPU")
        print("  2. 每个 GPU 独立前向传播")
        print("  3. 将各 GPU 的 loss 汇总到 GPU 0")
        print("  4. GPU 0 计算梯度并广播到所有 GPU")
        print("  5. 所有 GPU 用相同梯度更新参数")
        return

    # 假设我们有 4 张 GPU
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    # 简单模型
    model = nn.Linear(1024, 512).cuda()

    # ❌ 实际项目中不推荐: 用 DP 包装
    if n_gpus > 1:
        model = nn.DataParallel(model)
    else:
        print("Single GPU detected, DataParallel is a no-op wrapper.")

    # 模拟一次前向
    dummy_input = torch.randn(8, 1024).cuda()
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


# ============================================================
# DP 的致命缺陷总结 (在面试中可脱口而出)
# ============================================================
def explain_dp_issues():
    """总结 DP 的 4 个致命问题"""
    issues = [
        ("单卡瓶颈", "所有梯度汇集到 GPU 0, 由 GPU 0 负责 reduce + broadcast, 负载远高于其他卡"),
        ("Python GIL", "使用 Python 线程做多卡调度, GIL 成为瓶颈, 无法利用多核 CPU"),
        ("不支持多机", "只能在单机内使用, 无法扩展到多机多卡分布式场景"),
        ("通信效率低", "每个 batch 都要在 GPU 0 做 reduce, 无法与计算重叠, 延迟高"),
    ]
    print("=" * 60)
    print("DataParallel 致命缺陷清单:")
    print("=" * 60)
    for title, desc in issues:
        print(f"[{title}] {desc}")
    print("=" * 60)
    print("结论: 2026 年的所有分布式训练, 都应该用 DDP 替代 DP。")


if __name__ == "__main__":
    demo_dp()
    explain_dp_issues()
    print("OK")
