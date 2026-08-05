# ---
# chapter: 30
# topic: SFT、LoRA 与 QLoRA
# topic_id: llm_frameworks.llama_factory_dataset_prep
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: (none - pure stdlib)
# run: python 19_llama_factory_dataset_prep.py
# expected_runtime: <1s
# expected_output: dataset file preview
# ---
# See: ../../../30_SFT_LoRA与QLoRA.md
# Interview hooks:
#   1. Alpaca 格式与 ShareGPT 格式的差异是什么？如何选择？
#   2. 微调数据量需要多少？质量与数量哪个更重要？
"""
准备微调数据的标准格式
支持 Alpaca 格式和 ShareGPT 格式
"""

import json
import os

# Alpaca 格式（推荐用于指令微调）
alpaca_data = [
    {
        "instruction": "你是一个Python专家，请解释以下概念。",
        "input": "Python中的GIL是什么？",
        "output": "GIL（Global Interpreter Lock，全局解释器锁）是CPython中的一个互斥锁，它确保同一时刻只有一个线程执行Python字节码。这意味着在多核CPU上，CPU密集型的Python多线程程序无法利用多核优势。但I/O密集型任务仍然可以受益于多线程，因为I/O操作会释放GIL。\n\n解决方案：\n1. 使用多进程（multiprocessing）替代多线程\n2. 使用C扩展释放GIL\n3. 使用其他Python实现（如Jython、IronPython）",
        "system": "",
        "history": [],
    },
]

# ShareGPT 格式（适合对话数据）
sharegpt_data = [
    {
        "conversations": [
            {"from": "human", "value": "你好，请问Python的装饰器是什么？"},
            {
                "from": "gpt",
                "value": "装饰器是Python中一种特殊的语法，允许在不修改原函数代码的情况下增加额外功能...",
            },
            {"from": "human", "value": "能给我一个实际例子吗？"},
            {"from": "gpt", "value": "当然！比如@staticmethod、@classmethod就是内置装饰器..."},
        ],
        "system": "你是一个Python教学助手",
    }
]

# 保存为 JSON 文件
out_dir = "/tmp/ch18_demo"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "alpaca_data.json"), "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
with open(os.path.join(out_dir, "sharegpt_data.json"), "w", encoding="utf-8") as f:
    json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

print(f"Alpaca 样例数: {len(alpaca_data)}")
print(f"ShareGPT 样例数: {len(sharegpt_data)}")
print(f"已保存到: {out_dir}")

if __name__ == "__main__":
    print("OK")
