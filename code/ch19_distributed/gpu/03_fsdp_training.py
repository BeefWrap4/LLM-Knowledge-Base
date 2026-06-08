# ---
# chapter: 19
# topic: Fully Sharded Data Parallel (FSDP) 真实训练
# section: 19.3
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: torch, transformers, accelerate
# run: accelerate launch --num_processes 1 03_fsdp_training.py
#   或 (真实多卡) accelerate launch --num_processes 2 03_fsdp_training.py
# expected_runtime: 60-180s
# expected_output: FSDP 训练 loss 下降 + 显存节省统计
# ---
# See: ../tutorial/19_分布式训练系统.md §19.3
#
# Interview hooks:
#   1. FSDP vs DDP 核心区别？(答: FSDP 分片参数+梯度+优化器状态, 显存 N 倍节省; DDP 每 rank 完整副本)
#   2. FSDP forward/backward 流程？(答: all-gather params → compute → reduce-scatter grads)
#   3. FSDP wrapping 策略？(答: 按 transformer block wrap; auto_wrap_policy=TRANSFORMER_BASED_AUTO_WRAP_POLICY)
"""FSDP 完全分片数据并行训练 (真实 PyTorch FSDP).

单卡运行 (FSDP auto-shard):
  accelerate launch --num_processes 1 03_fsdp_training.py

多卡运行:
  accelerate launch --num_processes 2 03_fsdp_training.py

FSDP 核心: 参数 + 梯度 + 优化器状态 在 N 个 rank 间分片
  - 单卡: 内存优化等同 ZeRO-1
  - 多卡: 内存节省 N 倍 (按 N 线性)

注意: FSDP 在 Windows 上的 gloo 后端有兼容性 bug
  (hostname 包含中文/特殊字符时, gloo 无法解析网络设备).
  推荐在 Linux / WSL2 / Docker 内运行; Windows 用户建议在 WSL2 中跑本例.
"""

import os
import sys
from pathlib import Path

import torch

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    require_nvidia_gpu(min_vram_gb=8, min_count=1)


def _init_distributed(world_size: int):
    """初始化 FSDP 所需的 process group.

    - 单卡: accelerate launch --num_processes 1 不自动设 RANK, 手动补齐 + file:// init
    - 多卡: accelerate 自动设好 env, 用 nccl 后端
    """
    import torch.distributed as dist

    if dist.is_initialized():
        return

    if world_size == 1:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        # Windows gloo 走 env:// init_method 时会卡在 hostname 解析,
        # 用 file:// init_method 绕过. 但即使是 file://, gloo C++ 仍会
        # 调用 makeDeviceForHostname 解析本机网络设备, 在中文/特殊 hostname
        # 的 Windows 上会失败. 这是 PyTorch gloo v1 on Windows 的已知 bug.
        import tempfile

        init_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pt_init",
            dir=os.environ.get("TEMP", "/tmp"),
        )
        init_file.close()
        init_path = init_file.name.replace("\\", "/")
        dist.init_process_group(
            backend="gloo",
            init_method=f"file:///{init_path}",
            rank=0,
            world_size=1,
        )
    else:
        dist.init_process_group(backend="nccl")


def main():
    check_hardware()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 加载模型
    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(
            f"需要模型 {model_path}",
            "运行 `make download-models-default`.",
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 记录 FSDP 包装前显存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / (1024**3)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    vram_after_load = torch.cuda.memory_allocated() / (1024**3)

    # 尝试 init process group
    try:
        _init_distributed(world_size)
    except Exception as e:
        err_msg = str(e)
        # Windows gloo 已知 bug: 解析 hostname / 找网络设备失败
        if "makeDeviceForHostname" in err_msg or "makeDeviceForInterface" in err_msg:
            print("=" * 70)
            print("WARN: PyTorch gloo on Windows 无法解析本机网络设备")
            print(f"      错误: {err_msg[:200]}")
            print("      这是 PyTorch gloo v1 在 Windows + 中文/特殊 hostname 上的已知 bug.")
            print("      建议方案:")
            print("        1) 在 WSL2 / Linux 容器中跑: wsl -- bash 03_fsdp_training.py")
            print("        2) 多卡 Linux 节点: accelerate launch --num_processes N 03_fsdp_training.py")
            print("        3) FSDP 概念演示: 直接看 03_fsdp_concepts_demo.py (无需分布式)")
            print("=" * 70)
            print("\n[fallback] 进入 FSDP 概念演示 (无 distributed init)...")
            _demo_fsdp_concepts(model, model_path, vram_before, vram_after_load, device)
            return
        else:
            raise

    if local_rank == 0:
        print(f"FSDP 初始化: world_size={world_size}")

    # PyTorch 2.9+ 新签名: transformer_auto_wrap_policy 需要 partial 绑定
    # (FSDP 内部会传入 module/recurse/nonwrapped_numel 三个位置参数)
    from functools import partial

    from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
    )

    vram_after_fsdp = torch.cuda.memory_allocated() / (1024**3)

    # 合成 data
    torch.manual_seed(42)
    dataset = [
        {"input_ids": torch.randint(0, 1000, (64,)), "labels": torch.randint(0, 1000, (64,))}
        for _ in range(200)
    ]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 训练
    num_steps = 100
    model.train()
    losses = []
    step = 0
    for epoch in range((num_steps // len(loader)) + 1):
        for batch in loader:
            if step >= num_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if step % 10 == 0 and local_rank == 0:
                vram_now = torch.cuda.memory_allocated() / (1024**3)
                print(f"[rank {local_rank}] step {step:3d} | loss={loss.item():.4f} | vram={vram_now:.2f}GB")
            step += 1

    if local_rank == 0:
        avg_loss = sum(losses) / len(losses)
        vram_final = torch.cuda.memory_allocated() / (1024**3)
        vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
        print("\n=== FSDP 训练完成 ===")
        print(f"  total steps: {len(losses)}")
        print(f"  avg loss: {avg_loss:.4f}")
        print(f"  final loss: {losses[-1]:.4f}")
        print(f"  vram before FSDP: {vram_before:.2f}GB")
        print(f"  vram after FSDP:  {vram_after_fsdp:.2f}GB")
        print(f"  vram final:       {vram_final:.2f}GB")
        print(f"  vram peak:        {vram_peak:.2f}GB")
        if losses[-1] < losses[0]:
            print(f"  loss 下降: {losses[0]:.4f} -> {losses[-1]:.4f}")

    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


def _demo_fsdp_concepts(model, model_path: str, vram_before: float, vram_after_load: float, device):
    """FSDP 概念演示 (Windows gloo 不可用时的 fallback).

    不真跑 FSDP, 但展示:
      1) Qwen2.5-0.5B 的实际显存占用
      2) FSDP 在 1 卡 / N 卡下的理论节省
      3) 真实训练 loss 下降曲线 (用 DDP-less 训练, 走 FSDP 同一 model)
    """
    print("\n=== FSDP 概念演示 (Windows gloo 不可用 fallback) ===\n")
    print("[1/3] 加载 Qwen2.5-0.5B 测量实际参数显存...")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model: {model_path.split('/')[-1]}")
    print(f"  total params:     {n_params / 1e6:.1f}M")
    print(f"  trainable params: {n_trainable / 1e6:.1f}M")
    print(f"  vram before load: {vram_before:.2f}GB")
    print(f"  vram after load:  {vram_after_load:.2f}GB")
    print(f"  model weights:    ~{vram_after_load - vram_before:.2f}GB (BF16)")

    print("\n[2/3] FSDP 显存节省理论 (1 vs N 卡):")
    # 1 卡 = 不分片; N 卡 = 每个 rank 只持 1/N
    full_mem = n_params * 2 / 1e9  # BF16 = 2 bytes
    print(f"  无 FSDP (DDP): 每 rank 完整副本 = {full_mem:.2f}GB")
    print(f"  FSDP N=1:     {full_mem:.2f}GB  (不分片, 但可省 optimizer state)")
    print(f"  FSDP N=2:     {full_mem / 2:.2f}GB  (参数+grad+opt 各分 1/2)")
    print(f"  FSDP N=4:     {full_mem / 4:.2f}GB")
    print(f"  FSDP N=8:     {full_mem / 8:.2f}GB")
    # AdamW 状态: fp32 master + 2 momentum = 12 bytes/param
    adam_state_gb = n_params * 12 / 1e9
    print(f"  AdamW 状态 (全精度, BF16 模型): {adam_state_gb:.2f}GB")
    print(f"  实际节省: 约 {full_mem + adam_state_gb:.2f}GB → {(full_mem + adam_state_gb) / 8:.2f}GB (8 卡)")

    print("\n[3/3] 真实训练演示 (无 FSDP, 验证 Qwen2.5 可训练 + loss 下降):")
    # 真跑 30 step 训练, 证明 Qwen2.5-0.5B 在 5090D 上能 fine-tune
    # (model 已在 main() 阶段 to(device) 过了, 这里直接用)
    torch.manual_seed(42)
    dataset = [
        {"input_ids": torch.randint(0, 1000, (64,)), "labels": torch.randint(0, 1000, (64,))}
        for _ in range(60)
    ]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    losses = []
    for step, batch in enumerate(loader):
        if step >= 30:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if step % 5 == 0:
            print(f"  step {step:3d} | loss={loss.item():.4f}")

    print("\n  30 step 完成:")
    print(f"    initial loss: {losses[0]:.4f}")
    print(f"    final loss:   {losses[-1]:.4f}")
    if losses[-1] < losses[0]:
        print(f"    loss 下降: {losses[0]:.4f} -> {losses[-1]:.4f}")
    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"    vram peak:    {vram_peak:.2f}GB / 34.19GB")

    print("\n=== 关键 takeaway ===")
    print("  FSDP 在单卡/Windows 上的 runtime 限制是 PyTorch gloo 实现层,")
    print("  与 FSDP API 本身无关. 真正跑 FSDP 多卡分片的演示请在 WSL2/Linux 上跑.")
    print("  代码 100% 真实 FSDP API, 没有 mock; 只是在 Windows gloo 上不可执行.")


if __name__ == "__main__":
    main()
