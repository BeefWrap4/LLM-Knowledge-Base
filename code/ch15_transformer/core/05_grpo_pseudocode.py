# ---
# chapter: 31
# topic: 偏好对齐与强化学习
# topic_id: transformer.grpo_pseudocode
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: 纯 Python (无外部依赖); 可选 torch 用于 demo
# run: python 05_grpo_pseudocode.py
# expected_runtime: <3s (CPU)
# expected_output: 打印 GRPO 优势计算示例; 演示组内归一化效果
# ---
# See: ../../../31_偏好对齐与强化学习.md
# Interview hooks:
#   1. GRPO 相比 PPO 的核心创新是什么？为什么能"去 Critic 化"？
#   2. 组内相对优势 (group-relative advantage) 的标准化公式及其意义？
#   3. GRPO 为什么特别适合推理任务 (数学/编程等有明确答案的任务)？

# GRPO 伪代码 (来自教程)
# for batch in dataloader:
#     # 1. 对同一问题采样 G 个回答
#     responses = [generate(model, question) for _ in range(G)]
#
#     # 2. 奖励评分（可基于规则或奖励模型）
#     rewards = [reward_fn(q, r) for r in responses]
#
#     # 3. 计算组内相对优势
#     mean_reward = sum(rewards) / G
#     advantages = [r - mean_reward for r in rewards]
#
#     # 4. 策略梯度更新
#     loss = -sum(advantage * log_prob for advantage, log_prob in zip(advantages, log_probs))
#     loss.backward()


def grpo_advantages(rewards):
    """计算 GRPO 优势 (Group Relative Policy Optimization).

    公式: A_i = (r_i - mean({r_j})) / std({r_j})

    相比简单 "r - mean", 除以 std 让不同奖励量级的任务可比较,
    避免大批量/小批量更新的梯度方差差异过大。
    """
    import statistics

    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    mean_r = statistics.mean(rewards)
    std_r = statistics.pstdev(rewards)  # 总体标准差
    if std_r < 1e-8:
        # 所有回答得分相同 → 优势为 0 (无偏好信号)
        return [0.0 for _ in rewards]
    return [(r - mean_r) / std_r for r in rewards]


def grpo_loss(advantages, log_probs):
    """GRPO 策略梯度损失: -sum(A_i * log_prob_i)."""
    assert len(advantages) == len(log_probs), "长度不一致"
    return -sum(a * lp for a, lp in zip(advantages, log_probs))


def reward_fn_stub(question, response):
    """Mock 奖励函数: 演示用. 真实实现可基于规则或奖励模型.

    这里简单按响应长度近似, 让 demo 有非零优势信号.
    """
    return min(len(response) / 50.0, 1.0)


if __name__ == "__main__":
    # 模拟一个 batch: 一个问题, 4 个回答
    question = "求解方程 x^2 - 4 = 0"
    responses = [
        "x = 2",  # 短回答
        "x = 2 或 x = -2",  # 标准回答
        "x = ±2, 即 2 和 -2",  # 详细回答
        "x = sqrt(4) = 2, 另一个解是 -sqrt(4) = -2",  # 详细推导
    ]

    # 1) 对每个回答打分
    rewards = [reward_fn_stub(question, r) for r in responses]
    print("回答奖励:", [f"{r:.2f}" for r in rewards])

    # 2) 计算组内相对优势 (GRPO 核心)
    advantages = grpo_advantages(rewards)
    print("GRPO 优势:", [f"{a:+.3f}" for a in advantages])
    print("  # 正值 → 优于组平均, 应增加其概率")
    print("  # 负值 → 劣于组平均, 应降低其概率")
    print("  # 接近 0 → 接近组平均, 几乎不更新")

    # 3) 模拟策略给出的对数概率
    import math

    log_probs = [math.log(0.5), math.log(0.4), math.log(0.3), math.log(0.2)]
    loss = grpo_loss(advantages, log_probs)
    print(f"\nGRPO 损失: {loss:.4f}")
    print("  # 损失 < 0 表示高优势回答的 log_prob 较高 (好事)")

    # 4) 与 PPO 对比 (概念演示)
    # PPO 还需要:
    #   - critic 网络估计 V(s) → 优势 A_t = R_t - V(s_t)
    #   - 重要性采样比率 r_t(θ) = π_θ / π_old
    #   - clip 机制限制更新幅度
    # GRPO 通过"组内相对"完全替代 critic, 实现更简单
    print("\nPPO 需要的组件: Actor + Critic + Reward Model + Reference Model")
    print("GRPO 需要的组件: Actor + Reward Model(可选, 规则可替代)")
    print("  # 显存/参数节省 ~50% (无 Critic)")

    print("\nOK")
