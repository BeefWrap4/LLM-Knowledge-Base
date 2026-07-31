# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.3.3 HIL-SERL — 人在回路机器人强化学习
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: torch
# run: python 07_hil_serl.py
# expected_runtime: 5-15s (教学 critic 50 步训练)
# expected_output: TD loss 下降 + HIL-SERL replay/intervention 机制
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.3.3
#
# Interview hooks:
#   1. HIL-SERL 如何用人类干预纠正在线探索，而不是给“干预奖励”？
#   2. demo buffer 与在线 RL buffer 为什么要分开并做平衡采样？
#   3. 二元奖励分类器、人类干预与 RLPD learner 分别负责什么？
"""HIL-SERL 机制与教学 critic demo。

论文/官方项目的核心流程：
  1. 遥操作采集正负图像，离线训练二元任务成功分类器。
  2. 将少量专家演示放入 demo buffer。
  3. 在线 rollout 时，人类可接管并执行专家动作；所有 transition 进入在线 RL buffer，
     干预 transition 还会额外复制到 intervention/demo buffer。
  4. learner 从 demo/intervention buffer 与在线 RL buffer 平衡采样，使用 RLPD 更新策略。

重要边界：
  - 人类干预会覆盖策略动作，但论文没有把“发生干预”直接写成固定正奖励。
  - 本文件只训练一个简化 TD critic，用来说明 replay 和 action takeover；它不是完整
    HIL-SERL/RLPD 复现，未实现 actor、Q ensemble、熵项、视觉奖励分类器或机器人控制环。
  - 论文的训练时间与成功率属于其机器人、任务和实验设置，不是通用效率承诺。
"""

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu, skip_if_mock


def check_hardware() -> None:
    # 该合成教学网络很小，不虚构生产 HIL-SERL 的统一显存门槛。
    require_nvidia_gpu(min_vram_gb=0, min_count=1)


@dataclass(frozen=True)
class Transition:
    """一个 replay transition；reward 来自任务奖励/二元奖励分类器。"""

    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    intervened: bool
    source: Literal["demo", "online"]


def executed_action(
    policy_action: torch.Tensor,
    expert_action: torch.Tensor | None,
    *,
    intervened: bool,
) -> torch.Tensor:
    """返回机器人实际执行的动作。

    人类介入时专家动作覆盖策略动作；该信号不会在此函数中修改 reward。
    """
    if intervened:
        if expert_action is None:
            raise ValueError("intervened=True 时必须提供 expert_action")
        return expert_action.detach().clone()
    return policy_action.detach().clone()


def record_online_transition(
    *,
    state: torch.Tensor,
    policy_action: torch.Tensor,
    expert_action: torch.Tensor | None,
    classifier_reward: float,
    next_state: torch.Tensor,
    done: bool,
    intervened: bool,
) -> Transition:
    """记录在线 transition；分类器奖励与介入标志保持独立。"""
    return Transition(
        state=state.detach().clone(),
        action=executed_action(policy_action, expert_action, intervened=intervened),
        reward=float(classifier_reward),
        next_state=next_state.detach().clone(),
        done=done,
        intervened=intervened,
        source="online",
    )


def store_online_transition(
    transition: Transition,
    *,
    online_buffer: list[Transition],
    demo_or_intervention_buffer: list[Transition],
) -> None:
    """复现官方代码的 buffer 路由：全部进 online，干预样本再进 demo/intervention。"""
    if transition.source != "online":
        raise ValueError("store_online_transition 只接受在线 transition")
    online_buffer.append(transition)
    if transition.intervened:
        demo_or_intervention_buffer.append(transition)


def balanced_replay_batch(
    demo_or_intervention_buffer: list[Transition],
    online_buffer: list[Transition],
    *,
    per_buffer: int,
    seed: int = 0,
) -> list[Transition]:
    """从两个 buffer 各采 ``per_buffer`` 条，演示官方实现的 50/50 RLPD 采样。"""
    if per_buffer < 1:
        raise ValueError("per_buffer 必须大于 0")
    if len(demo_or_intervention_buffer) < per_buffer or len(online_buffer) < per_buffer:
        raise ValueError("两个 replay buffer 都必须包含足够样本")

    rng = random.Random(seed)
    return rng.sample(demo_or_intervention_buffer, per_buffer) + rng.sample(
        online_buffer,
        per_buffer,
    )


def collate_transitions(
    transitions: list[Transition],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """把教学 replay 样本堆叠成 critic batch。"""
    state = torch.stack([item.state for item in transitions]).to(device)
    action = torch.stack([item.action for item in transitions]).to(device)
    reward = torch.tensor([[item.reward] for item in transitions], device=device)
    next_state = torch.stack([item.next_state for item in transitions]).to(device)
    done = torch.tensor([[item.done] for item in transitions], dtype=torch.float32, device=device)
    return state, action, reward, next_state, done


class TeachingQNetwork(nn.Module):
    """教学用单 Q critic；完整 RLPD/SAC 使用更完整的 actor-critic 组件。"""

    def __init__(self, state_dim: int = 14, action_dim: int = 7, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


def teaching_critic_loss(
    q_net: TeachingQNetwork,
    target_q: TeachingQNetwork,
    state: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    next_action: torch.Tensor,
    done: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """简化 off-policy TD loss；不冒充完整 SAC/RLPD 目标。"""
    with torch.no_grad():
        target = reward + gamma * (1.0 - done) * target_q(next_state, next_action)
    prediction = q_net(state, action)
    return F.mse_loss(prediction, target)


def _synthetic_replay() -> tuple[list[Transition], list[Transition]]:
    """构造可离线解释的合成 replay；不是论文数据或 benchmark。"""
    demo_buffer: list[Transition] = []
    online_buffer: list[Transition] = []
    generator = torch.Generator().manual_seed(7)

    for idx in range(24):
        state = torch.randn(14, generator=generator)
        expert_action = torch.tanh(torch.randn(7, generator=generator))
        next_state = state + 0.05 * torch.randn(14, generator=generator)
        demo_buffer.append(
            Transition(
                state=state,
                action=expert_action,
                reward=float(idx % 3 == 0),
                next_state=next_state,
                done=idx % 3 == 0,
                intervened=False,
                source="demo",
            )
        )

    for idx in range(24):
        state = torch.randn(14, generator=generator)
        policy_action = torch.tanh(torch.randn(7, generator=generator))
        expert_action = torch.tanh(torch.randn(7, generator=generator))
        next_state = state + 0.05 * torch.randn(14, generator=generator)
        transition = record_online_transition(
            state=state,
            policy_action=policy_action,
            expert_action=expert_action,
            classifier_reward=float(idx % 5 == 0),
            next_state=next_state,
            done=idx % 5 == 0,
            intervened=idx % 4 == 0,
        )
        store_online_transition(
            transition,
            online_buffer=online_buffer,
            demo_or_intervention_buffer=demo_buffer,
        )
    return demo_buffer, online_buffer


def main() -> None:
    if skip_if_mock("NVIDIA GPU、PyTorch CUDA；真实机器人复现还需要遥操作和相机/机械臂"):
        return

    check_hardware()
    torch.manual_seed(7)
    device = torch.device("cuda")
    print("=== HIL-SERL 机制 + 教学 critic demo ===\n")

    demo_buffer, online_buffer = _synthetic_replay()
    batch = balanced_replay_batch(demo_buffer, online_buffer, per_buffer=16, seed=7)
    state, action, reward, next_state, done = collate_transitions(batch, device=device)
    next_action = torch.zeros_like(action)

    q_net = TeachingQNetwork().to(device)
    target_q = TeachingQNetwork().to(device)
    target_q.load_state_dict(q_net.state_dict())
    target_q.requires_grad_(False)
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=3e-4)

    print("  replay: 16 demo/intervention + 16 online（教学展示 50/50 采样）")
    print(f"  online 人类介入样本: {sum(item.intervened for item in online_buffer)} / 24")
    print("  reward: 来自合成二元任务标签，不因 intervened=True 自动加分")
    print("  critic: 单 Q TD 教学模型；不是完整 HIL-SERL/RLPD\n")

    losses: list[float] = []
    for step in range(50):
        loss = teaching_critic_loss(
            q_net,
            target_q,
            state,
            action,
            reward,
            next_state,
            next_action,
            done,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"  step {step:3d} | teaching TD loss = {loss.item():.4f}")

    print(f"\n  loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print("\nHIL-SERL 论文机制（Luo et al., 2024）:")
    print("  1. 遥操作正/负图像 → 训练二元任务成功分类器")
    print("  2. 少量专家演示 → demo/intervention buffer")
    print("  3. 在线 rollout：全部 transition 进 RL buffer；干预样本还会复制到 demo/intervention buffer")
    print("  4. learner 50/50 采样 demo/intervention 与在线 RL buffer，并使用 RLPD 更新")
    print("  5. 随策略成功率和周期时间改善，逐步减少介入")
    print("  6. 论文结果只适用于其任务、硬件和训练配置，不能外推固定倍数或时长")
    print("OK")


if __name__ == "__main__":
    main()
