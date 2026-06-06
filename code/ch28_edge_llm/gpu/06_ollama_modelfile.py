# ---
# chapter: 28
# topic: Ollama Modelfile 自定义模型
# section: 28.4.2 Ollama 一键部署
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: ollama (CLI)
# run: python 06_ollama_modelfile.py
# expected_runtime: <1s (生成 Modelfile 字符串, 不实际调用 ollama)
# expected_output: 多场景 Modelfile 模板 + Python 化封装
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.4.2
# Interview hooks:
#   1. Ollama Modelfile 相比直接用 GGUF 模型增加了哪些能力?
#   2. SYSTEM 指令在端侧 LLM 中如何影响行为?
#   3. 如何用 ADAPTER 字段加载 LoRA 微调权重?
"""Ollama Modelfile 模板生成器 - 用 Python 字符串构造而非手写."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Modelfile:
    """Ollama Modelfile 的 Python 化封装."""
    base_model: str
    system: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_ctx: int = 2048
    stop: list = field(default_factory=list)
    parameters_extra: list = field(default_factory=list)
    adapter: Optional[str] = None
    template: Optional[str] = None

    def render(self) -> str:
        """渲染为标准 Modelfile 文本."""
        lines = [f"FROM {self.base_model}"]
        lines.append("")

        if self.system:
            # 3 引号包裹多行 system
            lines.append('SYSTEM """')
            lines.append(self.system)
            lines.append('"""')
            lines.append("")

        if self.template:
            lines.append(f'TEMPLATE """{self.template}"""')
            lines.append("")

        lines.append("# PARAMETERS")
        lines.append(f"PARAMETER temperature {self.temperature}")
        lines.append(f"PARAMETER top_p {self.top_p}")
        lines.append(f"PARAMETER top_k {self.top_k}")
        lines.append(f"PARAMETER num_ctx {self.num_ctx}")
        for s in self.stop:
            lines.append(f"PARAMETER stop {s}")
        for p in self.parameters_extra:
            lines.append(f"PARAMETER {p}")

        if self.adapter:
            lines.append("")
            lines.append(f"ADAPTER {self.adapter}")

        return "\n".join(lines)


def example_chatbot() -> None:
    """示例 1: 中文对话助手."""
    mf = Modelfile(
        base_model="llama3.2:3b",
        system="你是一个友好的中文助手, 回答简洁, 不超过 200 字.",
        temperature=0.7,
        num_ctx=4096,
    )
    print("--- 示例 1: 中文聊天 ---")
    print(mf.render())


def example_coder() -> None:
    """示例 2: 代码助手 + 停止符."""
    mf = Modelfile(
        base_model="qwen2.5-coder:7b",
        system="You are an expert Python developer. Always include type hints.",
        temperature=0.2,  # 代码: 低温度更稳定
        stop=["```", "\n# >>>"],
        parameters_extra=["repeat_penalty 1.1"],
    )
    print("\n--- 示例 2: 代码助手 ---")
    print(mf.render())


def example_with_lora() -> None:
    """示例 3: 加载 LoRA 微调权重."""
    mf = Modelfile(
        base_model="llama3.2:3b",
        system="你是公司内部客服, 严格按照产品手册回答.",
        adapter="./lora-customer-service-q4.gguf",
        temperature=0.3,
    )
    print("\n--- 示例 3: LoRA 微调 ---")
    print(mf.render())


def main() -> None:
    example_chatbot()
    example_coder()
    example_with_lora()
    print("\n💡 使用方法: ollama create my-model -f Modelfile && ollama run my-model")


if __name__ == "__main__":
    main()
    print("OK")
