# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.6.1 世界模型 - NVIDIA Cosmos 未来帧预测
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: torch (真实模式需 cosmos + diffusers)
# run: python 08_cosmos_world_model.py
# expected_runtime: <5s (mock)
# expected_output: 演示 world model 的输入/输出形状
# ---
# See: ../tutorial/21_多模态大模型.md#21-6-1-世界模型（world-models）
# Interview hooks:
#   1. 世界模型 (World Model) 与普通视频生成模型的核心区别是什么？
#   2. Cosmos 在机器人 / 自动驾驶场景中如何接入 RL 训练？
#   3. guidance_scale 在条件视频生成中起什么作用？

import os


def main():
    use_mock = os.environ.get("CH21_MOCK", "1") == "1"

    if use_mock:
        # 演示世界模型的输入/输出张量形状
        import torch

        B, T_in, T_out, H, W = 1, 4, 64, 256, 256
        action_dim = 7
        # 当前观测 (T_in 帧历史)
        current_obs = torch.randn(B, T_in, H, W, 3)
        # 机器人动作序列
        robot_action_seq = torch.randn(B, T_out, action_dim)
        # 模拟未来帧预测输出
        future_frames = torch.randn(B, T_out, H, W, 3)
        print(f"Input obs shape:    {tuple(current_obs.shape)}")
        print(f"Input action shape: {tuple(robot_action_seq.shape)}")
        print(f"Output frames shape: {tuple(future_frames.shape)}")
        print("Cosmos world model demo OK")
        return

    # 真实模式：使用 NVIDIA Cosmos 推理
    try:
        from cosmos import CosmosPredictPipeline
    except ImportError:
        print("cosmos package not installed. Skipping real mode.")
        return

    pipe = CosmosPredictPipeline.from_pretrained(
        "nvidia/Cosmos-1.0-Diffusion-7B-Video2World"
    )
    future_frames = pipe(
        init_frames=current_obs,           # [B, T_in, H, W, 3]
        actions=robot_action_seq,           # [B, T_out, action_dim]
        num_future_frames=64,
        guidance_scale=7.0,
    ).frames
    print(f"Generated future frames shape: {tuple(future_frames.shape)}")


if __name__ == "__main__":
    main()
    print("OK")
