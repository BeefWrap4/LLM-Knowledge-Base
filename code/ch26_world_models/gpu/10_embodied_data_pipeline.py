# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.5 具身数据工程 — 仿真 + 真实 + 视频预训练
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: (无强依赖 — lerobot 可选, 缺则用简化实现)
# run: python 10_embodied_data_pipeline.py
# expected_runtime: 5-10s (合成数据生成 + LeRobot 格式示例)
# expected_output: LeRobot 数据 schema + episode 帧样本
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.5
#
# Interview hooks:
#   1. LeRobot 数据格式的核心 schema? (parquet per episode + meta JSON + stats)
#   2. 真实机器人数据如何归一化? (mean/std per feature, 存于 stats.safetensors)
#   3. 仿真数据 (Isaac) 与真实数据如何混合? (域随机化 + 真实比例 1:3)
"""具身数据 pipeline 演示 (LeRobot dataset 加载 + 转换).

数据格式: LeRobotDataset (v2.0+)
  - data/chunk-{N:03d}/episode_{M:06d}.parquet: 每 episode 一表
  - meta/info.json: fps, robot_type, features
  - meta/stats.safetensors: 各 feature 的 mean/std (归一化用)
  - meta/episodes.jsonl: episode 元数据

本 demo: 简化 LeRobot dataset schema + 生成合成 Aloha 数据样本.
生产: 加载 HF Hub 上的 lerobot/* 数据集 (e.g. lerobot/aloha_sim_transfer_cube_human).
"""

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))


@dataclass
class EpisodeSample:
    """单帧样本 (LeRobot schema 简化版)."""

    timestamp: float
    episode_index: int
    frame_index: int
    state: list  # 14-DoF 双臂状态
    action: list  # 7-DoF 末端动作
    image_shape: tuple = (3, 480, 640)  # 占位, 真实为 PNG/JPG bytes
    language_instruction: str = ""


class LeRobotLikeDataset:
    """LeRobot 兼容的简化 dataset (不依赖 lerobot 库).

    生产: from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    """

    def __init__(self, repo_id: str = "lerobot/aloha_sim_transfer_cube_human"):
        self.repo_id = repo_id
        self.fps = 50
        self.robot_type = "aloha"
        self.features = {
            "observation.state": {"shape": (14,), "dtype": "float32", "names": ["joint_pos"] * 14},
            "observation.images.top": {"shape": (3, 480, 640), "dtype": "uint8"},
            "observation.images.wrist": {"shape": (3, 480, 640), "dtype": "uint8"},
            "action": {
                "shape": (7,),
                "dtype": "float32",
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            },
        }
        self.n_episodes = 50
        self.episode_lengths = [200] * self.n_episodes  # 4 秒 @ 50Hz

    def __len__(self) -> int:
        return sum(self.episode_lengths)

    @property
    def total_frames(self) -> int:
        return len(self)

    @property
    def total_duration_s(self) -> float:
        return self.total_frames / self.fps

    def __iter__(self) -> Iterator[EpisodeSample]:
        rng_state = 0
        for ep_idx in range(self.n_episodes):
            # 简化的轨迹: state 缓慢变化 + 周期性 action
            for step in range(self.episode_lengths[ep_idx]):
                t = ep_idx * self.episode_lengths[0] + step
                phase = (t / self.fps) * 2 * 3.14159
                state = [0.1 * (i + 1) * (0.5 + 0.1 * (i * t % 7)) for i in range(14)]
                action = [0.05 * (1 + i) * (0.3 + 0.1 * (i * (t + 1) % 5)) for i in range(7)]
                yield EpisodeSample(
                    timestamp=t / self.fps,
                    episode_index=ep_idx,
                    frame_index=step,
                    state=state,
                    action=action,
                    language_instruction="pick up the cube and place it in the bin",
                )


def print_dataset_info(dataset: LeRobotLikeDataset) -> None:
    print(f"Dataset: {dataset.repo_id}")
    print(f"  robot_type : {dataset.robot_type}")
    print(f"  fps        : {dataset.fps}")
    print(f"  n_episodes : {dataset.n_episodes}")
    print(f"  total frames: {dataset.total_frames:,}")
    print(f"  total time : {dataset.total_duration_s:.1f} s")
    print("  features   :")
    for k, v in dataset.features.items():
        print(f"    {k:30s} {v['shape']} {v['dtype']}")


def main() -> None:
    print("=== 具身数据 pipeline (LeRobot 兼容 schema) ===\n")
    dataset = LeRobotLikeDataset()
    print_dataset_info(dataset)
    print()

    # 演示: 拉取前 3 帧
    print("步骤 1: 流式遍历前 3 帧 (生产用 LeRobotDataset[0] 随机访问)")
    samples = []
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        samples.append(sample)
        print(f"  ep={sample.episode_index} frame={sample.frame_index} t={sample.timestamp:.3f}s")
        print(f"    state[0:3]  = {[round(x, 3) for x in sample.state[:3]]}")
        print(f"    action[0:3] = {[round(x, 3) for x in sample.action[:3]]}")
    print()

    # 演示: episode 级元数据
    print("步骤 2: Episode 元数据 (生产存于 meta/episodes.jsonl)")
    meta = {
        "repo_id": dataset.repo_id,
        "fps": dataset.fps,
        "robot_type": dataset.robot_type,
        "total_episodes": dataset.n_episodes,
        "total_frames": dataset.total_frames,
        "features": dataset.features,
    }
    print(f"  meta: {json.dumps({k: v for k, v in meta.items() if k != 'features'}, indent=2)}")

    # 演示: 归一化所需 stats
    print("\n步骤 3: 归一化统计（文件名与 schema 以当前数据格式为准）")
    print("  概念: stats = compute_episode_stats(dataset)")
    print("    normalized_state = (state - mean) / std")
    print("    normalized_action = (action - mean) / std")
    print("  作用: 训练时把 state/action 标准化到 N(0, 1), 提升优化稳定性")

    print()
    print("=" * 60)
    print("真实数据管线接入边界:")
    print("  - LeRobotDataset import 路径、feature schema 与 repo_id 按当前发行版文档核对")
    print("  - 仿真任务名、启动器和导出格式按当前 Isaac Lab 任务注册表核对")
    print("  - 转换后校验时间戳、episode 边界、观测/动作单位、缺帧与数据 revision")
    print("  - 本脚本只处理内存中的合成 episode，未下载或转换外部数据")


if __name__ == "__main__":
    main()
    print("OK")
