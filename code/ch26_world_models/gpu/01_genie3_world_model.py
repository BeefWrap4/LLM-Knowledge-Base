# ---
# chapter: 26
# topic: 世界模型与具身AI
# section: 26.4.1 Genie 3 — Google 可交互世界模型
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: transformers, torch
# run: python 01_genie3_world_model.py
# expected_runtime: 30-90s (Qwen2.5-0.5B 真实加载 + 3 步 rollout)
# expected_output: 文本世界模型 3 步预测演示
# ---
# See: ../tutorial/26_世界模型与具身AI.md §26.4.1
#
# Interview hooks:
#   1. Genie 3 与 Cosmos 在训练目标上有什么不同？（可交互性 vs 物理一致性）
#   2. 世界模型核心公式 s_{t+1}, r_t = f(s_t, a_t) 如何用 LLM 模拟？
#   3. 用 LLM 做世界模型的优缺点：泛化强但物理一致性弱？
"""世界模型 (Genie 3 / Cosmos 类) 演示.

Genie 3 是 Google DeepMind 的可交互世界模型, 无开源权重.
此处用 Qwen2.5-0.5B 作为文本世界模型 (想象式环境预测) 替代演示.

世界模型核心:
  s_{t+1}, r_t = f(s_t, a_t)   # 状态 + 动作 → 下一状态 + 奖励

Genie 3 真实场景: 视频帧 + 动作输入 → 下一帧视频.
本 demo 简化: 文本描述 + 文本动作 → 下一状态描述.
"""

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


class TextWorldModel:
    """文本世界模型: 状态 = 文本描述, 动作 = 文本指令.

    用 LLM 作为世界模型, 输入 "state | action", 输出 "next_state".
    简化: 用 Qwen2.5-0.5B-Instruct 演示 f(s,a) → s' 的近似.
    """

    WORLD_MODEL_PROMPT = (
        "You are a text world model. Given a current state and an action, "
        "predict the next state in one short sentence (max 20 words). "
        "If the action succeeds, start the next state with 'success: '. "
        "Otherwise just describe the new state.\n\n"
    )

    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def step(self, state: str, action: str, max_new_tokens: int = 48) -> tuple[str, float]:
        """预测 next_state 并计算 reward (1.0 = success, 0.0 = otherwise)."""
        user_msg = f"State: {state}\nAction: {action}\nNext state:"
        messages = [
            {"role": "system", "content": self.WORLD_MODEL_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs.input_ids.size(1) :]
        next_state = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # 移除 LLM 可能复述的 "Next state:" 前缀
        next_state = next_state.replace("Next state:", "").strip()
        reward = 1.0 if next_state.lower().startswith("success") else 0.0
        return next_state, reward


def main() -> None:
    check_hardware()
    model_path = str(_code_root / "models" / "Qwen2.5-0.5B-Instruct")
    if not Path(model_path).exists():
        raise_with_help(
            f"需要 {model_path}",
            "运行 `make download-models-default` 或手动从 HuggingFace 下载 Qwen2.5-0.5B-Instruct.",
        )

    print("=== 文本世界模型 (Qwen2.5-0.5B 替代 Genie 3) ===\n")
    print("核心: s_{t+1}, r_t = f(s_t, a_t)  — 文本版近似")
    print(f"模型: {model_path}\n")

    wm = TextWorldModel(model_path)
    print(f"✅ 模型加载完成 (VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB)\n")

    # 演示 3 步 rollout: 客厅 → 走过去 → 拿起书 → 打开书
    state = "The living room is bright. There is a book on the sofa."
    actions = ["walk to the sofa", "pick up the book", "open the book"]

    for step, action in enumerate(actions):
        print(f"--- step {step} ---")
        print(f"  state  : {state}")
        print(f"  action : {action}")
        next_state, reward = wm.step(state, action)
        print(f"  next   : {next_state}")
        print(f"  reward : {reward}")
        print()
        state = next_state

    print("=" * 60)
    print("Genie 3 真实部署:")
    print("  - 输入: 视频帧 + 用户动作 (按键/摇杆)")
    print("  - 输出: 下一帧视频 (3D 一致性 + 可交互)")
    print("  - 训练: 大规模未标注视频 + 自动编码 latent action")
    print("  - 模型规模: 数十亿参数, 需数十 GB VRAM")
    print()
    print("本 demo 局限: 文本世界模型物理一致性弱, 需 video world model 替代.")


if __name__ == "__main__":
    main()
