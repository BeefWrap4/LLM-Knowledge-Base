# ---
# chapter: 23
# topic: AI安全与伦理
# section: 23.2 Prompt Injection与防御 - Token走私示例
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: none
# run: python 01_token_smuggling_demo.py
# expected_runtime: <1s
# expected_output: "OK" + 演示输出
# ---
# See: ../tutorial/23_AI安全与伦理.md#232-prompt-injection与防御-
# Interview hooks:
#   1. 什么是Token走私（Token Smuggling）？它如何绕过基于关键词的内容过滤？
#   2. 在Unicode层面有哪些常见的"隐形"字符可被用于注入攻击？
#   3. 如何检测和防御此类基于零宽字符的注入？
"""
Token走私（Token Smuggling）演示

攻击者利用特殊Unicode字符（零宽空格、零宽连接符等）绕过内容过滤。
视觉上看起来正常，但可以绕过基于关键词的过滤。

面试要点：
1. 理解Unicode特殊字符的攻击面
2. 掌握防御此类攻击的字符规范化方法
"""

import unicodedata
import re


def detect_invisible_chars(text: str) -> dict:
    """检测文本中的不可见/零宽字符"""
    invisible = []
    for i, ch in enumerate(text):
        # 零宽空格 (U+200B), 零宽非连接符 (U+200C), 零宽连接符 (U+200D)
        # 从左到右标记 (U+200E), 从右到左标记 (U+200F)
        if ch in ('​', '‌', '‍', '‎', '‏'):
            invisible.append({
                "position": i,
                "char": ch,
                "name": unicodedata.name(ch, "UNKNOWN"),
                "codepoint": f"U+{ord(ch):04X}"
            })
    return {
        "has_invisible": len(invisible) > 0,
        "count": len(invisible),
        "details": invisible
    }


def normalize_text(text: str) -> str:
    """字符规范化：去除零宽字符与控制字符"""
    # NFKC 规范化（兼容性分解再重组）
    text = unicodedata.normalize('NFKC', text)
    # 移除零宽字符
    text = re.sub(r'[​-‏‪-‮⁠-⁤]', '', text)
    return text


if __name__ == "__main__":
    # 演示：正常文本 vs 包含零宽字符的注入文本
    normal = "请帮我翻译这段文字"
    smuggled = "请帮我翻译这段文字"  # 中间包含零宽字符

    print("=== Token走私演示 ===")
    print(f"原始字符数: {len(normal)}")
    print(f"含零宽字符数: {len(smuggled)}")
    print(f"视觉一致: {normal.replace(' ', '') == smuggled.replace(' ', '')}")

    result = detect_invisible_chars(smuggled)
    print(f"检测到 {result['count']} 个不可见字符")

    normalized = normalize_text(smuggled)
    print(f"规范化后字符数: {len(normalized)}")
    print("OK")
