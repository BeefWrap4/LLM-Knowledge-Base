# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.5.3 Agent Trajectory 数据格式（ReAct 风格）
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: stdlib only
# run: python 09_agent_trajectory_sample.py
# expected_runtime: <1s
# expected_output: 序列化的 Agent 轨迹 JSON（含 thought/action/observation）
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. ReAct 风格 Agent 轨迹的核心结构是什么？Thought/Action/Observation 各自的作用？
#   2. 轨迹数据中如何处理"错误恢复"和"工具调用参数错误"以提升模型鲁棒性？
#   3. Trajectory 数据中 PII/API Key 脱敏为什么比纯文本更复杂？

import json

# Agent Trajectory 数据格式示例（ReAct 风格）
trajectory_sample = {
    "task_id": "weather_search_001",
    "instruction": "查询北京明天的天气并推荐合适的穿衣方案",
    "available_tools": [
        {"name": "get_weather", "params": ["city", "date"]},
        {"name": "search_clothing_advice", "params": ["temperature", "weather_condition"]}
    ],
    "trajectory": [
        {
            "step": 1,
            "thought": "用户想知道北京明天的天气和穿衣建议。先调用天气API获取数据。",
            "action": {
                "tool": "get_weather",
                "params": {"city": "北京", "date": "2026-06-07"}
            },
            "observation": {
                "temperature": "18-25°C",
                "condition": "晴间多云",
                "wind": "微风"
            }
        },
        {
            "step": 2,
            "thought": "温度在18-25度之间，需要搜索合适的穿衣建议。",
            "action": {
                "tool": "search_clothing_advice",
                "params": {"temperature": "18-25", "weather_condition": "晴间多云"}
            },
            "observation": {
                "advice": "建议穿着薄外套+T恤+长裤"
            }
        },
        {
            "step": 3,
            "thought": "已经获取了所有需要的信息，可以给出最终回复。",
            "action": "FINAL_ANSWER",
            "answer": "北京明天天气晴间多云，温度18-25°C，微风。建议穿着薄外套+T恤+长裤的搭配，灵活应对早晚温差。"
        }
    ],
    "outcome": "success",
    "num_steps": 3,
    "total_tokens": 1250
}


def main():
    # 校验轨迹一致性
    steps = trajectory_sample["trajectory"]
    assert trajectory_sample["num_steps"] == len(steps), "num_steps 不一致"
    for i, step in enumerate(steps, 1):
        assert step["step"] == i, f"step 编号不连续: {i} vs {step['step']}"
    print("轨迹一致性校验通过")
    print(f"任务: {trajectory_sample['instruction']}")
    print(f"步数: {trajectory_sample['num_steps']}, 结果: {trajectory_sample['outcome']}")
    print("\n完整轨迹 JSON:")
    print(json.dumps(trajectory_sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    print("OK")
