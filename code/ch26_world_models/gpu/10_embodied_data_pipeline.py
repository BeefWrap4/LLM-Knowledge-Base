# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.5 具身数据工程 — 仿真 + 真实 + 视频预训练
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: numpy, PIL (lazy)
# run: MOCK_MODE=1 python 10_embodied_data_pipeline.py
# expected_runtime: <3s
# expected_output: 异构数据 schema 统一、LeRobotDataset 写入摘要、统计
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.5
# Interview hooks:
#   1. VLA 训练数据三大来源（真实、仿真、视频）的优缺点？配比经验？
#   2. LeRobotDataset 的 Parquet + MP4 存储格式相对 HDF5 的优势？
#   3. 时间对齐 (timestamp sync) 在多相机 + 力矩 + 触觉数据上有什么坑？

"""
具身数据工程 —— 多源异构数据统一 pipeline。

三大数据源：
  A) 真实机器人遥操作 (Franka / ALOHA)
  B) 仿真数据 (Isaac Sim / MuJoCo / ManiSkill)
  C) 视频预训练 (Ego4D / Something-Something / Epic Kitchens)

输出统一 schema (LeRobot v2.0)：
  - frames  : 高频 (30Hz) 视频 + 触觉 → MP4 / 安全序列化
  - states  : 低频 (10Hz) 关节角、夹爪、力矩 → Parquet
  - actions : 低频 (10Hz) 目标关节角、夹爪开合 → Parquet
  - tasks   : 自然语言指令 → Parquet
"""

import os
import json
import math
import time
import numpy as np


MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"


# ---------- 1. 统一 Schema ----------
UNIFIED_SCHEMA = {
    "frame_id":        "int64",        # 全局连续编号
    "episode_id":      "int32",        # 任务轨迹编号
    "timestamp":       "float32",      # 相对 episode 起点的秒
    "observation.image.front":    "uint8[H,W,3] @ 30Hz",
    "observation.image.wrist":    "uint8[H,W,3] @ 30Hz",
    "observation.state":          "float32[D_state] @ 10Hz",  # 关节角
    "observation.velocity":       "float32[D_state] @ 10Hz",
    "observation.tactile":        "float32[N_taxels] @ 30Hz",
    "action":                     "float32[D_action] @ 10Hz",
    "action.is_intervention":     "bool @ 10Hz",  # HIL-SERL 接管信号
    "task":                       "string",  # 任务自然语言
    "source":                     "string",  # "real" | "sim" | "video_pretrain"
}


def print_schema() -> None:
    print("[Unified Schema] LeRobot v2.0 — 异构数据统一表示:")
    for k, v in UNIFIED_SCHEMA.items():
        print(f"  {k:38s} : {v}")
    print()


# ---------- 2. 多源 episode 适配器 ----------
class EpisodeAdapter:
    """把不同来源的 episode 转成统一 schema。"""

    def __init__(self, source: str, hz_video: int = 30, hz_control: int = 10):
        self.source = source
        self.hz_v = hz_video
        self.hz_c = hz_control
        self.dt_v = 1.0 / hz_video
        self.dt_c = 1.0 / hz_control

    def adapt(self, raw_episode: dict) -> dict:
        """raw_episode 结构因 source 而异；这里做归一化。"""
        if self.source == "real":
            return self._adapt_real(raw_episode)
        elif self.source == "sim":
            return self._adapt_sim(raw_episode)
        elif self.source == "video_pretrain":
            return self._adapt_video(raw_episode)
        else:
            raise ValueError(f"unknown source: {self.source}")

    def _adapt_real(self, ep: dict) -> dict:
        # 真实遥操作：含 2 路 wrist camera + 力矩
        return {
            "frames": ep["video"],            # (T_v, 2, H, W, 3)
            "states": ep["joints"],           # (T_c, D)
            "actions": ep["cmd"],             # (T_c, D)
            "tactile": ep.get("tactile"),     # (T_v, N)
            "task": ep["task"],
            "source": "real",
            "duration_s": ep["video"].shape[0] * self.dt_v,
        }

    def _adapt_sim(self, ep: dict) -> dict:
        # Isaac Sim：完美 ground-truth，但 sim2real gap
        return {
            "frames": ep["rgb"],              # (T_v, 1, H, W, 3)
            "states": ep["qpos"],             # (T_c, D)
            "actions": ep["target_qpos"],     # (T_c, D)
            "tactile": None,
            "task": ep["task"],
            "source": "sim",
            "duration_s": ep["rgb"].shape[0] * self.dt_v,
        }

    def _adapt_video(self, ep: dict) -> dict:
        # 视频预训练：只有图像，需要 inverse dynamics 提取伪 action
        return {
            "frames": ep["video"],
            "states": None,
            "actions": None,
            "tactile": None,
            "task": ep["task_text"],
            "source": "video_pretrain",
            "duration_s": ep["video"].shape[0] * self.dt_v,
        }


# ---------- 3. 简易 LeRobot-style Writer ----------
class LeRobotDatasetWriter:
    """Parquet (states/actions) + 元数据 JSON 模拟写入。"""

    def __init__(self, root: str = "/tmp/mock_lerobot_dataset"):
        self.root = root
        self.episodes = []
        self.total_frames = 0

    def add_episode(self, episode: dict) -> None:
        if MOCK_MODE:
            n_frames = int(episode["frames"].shape[0]) if episode.get("frames") is not None else 0
            self.total_frames += n_frames
            self.episodes.append({
                "source": episode["source"],
                "task": episode["task"],
                "duration_s": round(episode["duration_s"], 2),
                "n_frames": n_frames,
            })

    def summary(self) -> dict:
        if not self.episodes:
            return {}
        sources = {}
        for ep in self.episodes:
            sources[ep["source"]] = sources.get(ep["source"], 0) + 1
        return {
            "n_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "sources": sources,
            "total_duration_s": round(sum(e["duration_s"] for e in self.episodes), 2),
        }


# ---------- 4. 模拟 pipeline ----------
def demo_pipeline():
    rng = np.random.default_rng(42)
    writer = LeRobotDatasetWriter()

    # 真实遥操作 5 个 episode
    real_adapter = EpisodeAdapter(source="real")
    for i in range(5):
        ep = {
            "video": rng.integers(0, 255, (90, 2, 224, 224, 3), dtype=np.uint8),  # 3s
            "joints": rng.standard_normal((30, 7)).astype(np.float32),
            "cmd":   rng.standard_normal((30, 7)).astype(np.float32),
            "tactile": rng.standard_normal((90, 16)).astype(np.float32),
            "task": f"real_episode_{i}",
        }
        writer.add_episode(real_adapter.adapt(ep))

    # 仿真 50 个 episode（量大、便宜）
    sim_adapter = EpisodeAdapter(source="sim")
    for i in range(50):
        ep = {
            "rgb": rng.integers(0, 255, (60, 1, 224, 224, 3), dtype=np.uint8),
            "qpos": rng.standard_normal((20, 7)).astype(np.float32),
            "target_qpos": rng.standard_normal((20, 7)).astype(np.float32),
            "task": f"sim_episode_{i}",
        }
        writer.add_episode(sim_adapter.adapt(ep))

    # 视频预训练 1000 个 episode（Ego4D 风格）
    video_adapter = EpisodeAdapter(source="video_pretrain")
    for i in range(1000):
        ep = {
            "video": rng.integers(0, 255, (60, 1, 224, 224, 3), dtype=np.uint8),  # 2s
            "task_text": f"video_pretrain_clip_{i}",
        }
        writer.add_episode(video_adapter.adapt(ep))

    return writer.summary()


# ---------- 5. 训练配比分析 ----------
def mixing_ratio_analysis(stats: dict) -> None:
    total = stats["n_episodes"]
    print("\n[Mixing ratio] 经验配比（GR00T N1.5 / Pi0.5 paper）:")
    real_pct = stats["sources"].get("real", 0) / total * 100
    sim_pct  = stats["sources"].get("sim", 0) / total * 100
    vid_pct  = stats["sources"].get("video_pretrain", 0) / total * 100
    print(f"  real:           {stats['sources'].get('real', 0):4d} ep  ({real_pct:5.1f}%)")
    print(f"  sim:            {stats['sources'].get('sim', 0):4d} ep  ({sim_pct:5.1f}%)")
    print(f"  video_pretrain: {stats['sources'].get('video_pretrain', 0):4d} ep  ({vid_pct:5.1f}%)")
    print(f"\n  -> Real 数据稀少但 gold；Sim 量大但 sim2real gap；")
    print(f"     Video 极多但需 inverse dynamics 提取伪 action。")


# ---------- main ----------
def main() -> None:
    print("=== Embodied Data Pipeline (LeRobot v2.0) ===\n")
    print_schema()
    t0 = time.time()
    stats = demo_pipeline()
    dt = time.time() - t0
    print(f"[Pipeline] ingested 1055 episodes in {dt*1000:.0f}ms (mock)")
    print(f"[Pipeline] summary: {stats}")
    mixing_ratio_analysis(stats)
    print()


if __name__ == "__main__":
    main()
