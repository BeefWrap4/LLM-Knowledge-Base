# ---
# chapter: 46
# topic: 端侧、浏览器与边缘 LLM
# topic_id: lora_qlora.feature_router
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: (stdlib only)
# run: python 10_feature_router.py
# expected_runtime: <1s
# expected_output: 5 条 query 的 features dict + 路由目标
# ---
# See: ../../../46_端侧浏览器与边缘LLM.md
#
# Interview hooks:
#   1. 基于规则的路由 vs 基于分类器路由的优缺点？
#   2. 默认 tier 设为 cloud 而不是 device 的工程原因 (fail-safe)？
#   3. 如何用一个小分类器替代关键词匹配？训练数据如何收集？
"""特征路由器: 提取 query 特征, 路由到不同专家模型.

工程实践:
  - 启发式特征 (regex) — 上手快, 无需训练数据, 适合 v0
  - 小分类器 (3-5M 参数) — 准确率更高, 需 1k-10k 标注 query
  - 两者结合: 启发式做 high-confidence 路由, 分类器覆盖模糊 case
"""

import re
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import require_nvidia_gpu


def check_hardware():
    """此例子仅做特征提取 + 路由决策, 不加载模型; 但保持接口统一."""
    require_nvidia_gpu(min_vram_gb=0, min_count=1)


def extract_features(query: str) -> dict:
    """提取 query 启发式特征."""
    return {
        "length": len(query),
        "has_code": bool(re.search(r"```|def |class |import |function\s+\w+\s*\(", query)),
        "has_math": bool(re.search(r"[0-9]+[\+\-\*\/=]|[∑∏∫√]|\\frac|\\sum", query)),
        "is_question": query.strip().endswith("?") or query.strip().endswith("？"),
        "language": "zh" if re.search(r"[一-鿿]", query) else "en",
        "has_privacy": bool(re.search(r"身份证|住址|电话|银行卡|password|ssn", query, re.I)),
    }


def route(features: dict) -> str:
    """根据特征路由到不同专家模型 (fail-safe: 默走 cloud)."""
    if features["has_privacy"]:
        return "on-device (Qwen2.5-0.5B)"
    if features["has_code"]:
        return "code-expert (Qwen2.5-Coder-7B-Instruct)"
    if features["has_math"]:
        return "math-expert (Qwen2.5-Math-7B-Instruct)"
    if features["language"] == "zh":
        return "chinese-general (Qwen2.5-7B-Instruct)"
    return "cloud-general (Llama-3.3-70B)"


def main():
    check_hardware()

    print("=== 特征路由器 ===\n")
    queries = [
        "解释 Python 装饰器的用法",
        "Solve x^2 + 5x + 6 = 0",
        "How does the immune system work?",
        "Write a function to sort a list in Python",
        "我的身份证号 110101199001011234 帮我存一下",
    ]

    for q in queries:
        f = extract_features(q)
        r = route(f)
        print(f"Q: {q}")
        print(f"  features: {f}")
        print(f"  → 路由: {r}\n")
    print("OK")


if __name__ == "__main__":
    main()
