# ---
# chapter: 29
# topic: 大模型数据工程
# topic_id: data_engineering.back_translation
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: stdlib only
# run: python 06_back_translation.py
# expected_runtime: <2s
# expected_output: 原始文本与回译增强后的文本
# ---
# See: ../../../29_大模型数据工程.md
#
# Interview hooks:
#   1. 回译 (Back Translation) 数据增强的核心假设是什么？它在什么场景下会失效？
#   2. 与 EDA (Easy Data Augmentation) 相比，回译在 SFT 数据上有什么优劣？
#   3. 如何设计过滤机制避免回译引入噪声？


def back_translation_augment(
    texts: list[str],
    translate_func,  # 翻译函数 (text, source_lang, target_lang) -> translated_text
    source_lang: str = "zh",
    pivot_lang: str = "en",
    num_variants: int = 2,
) -> list[str]:
    """
    通过回译生成数据增强变体

    Args:
        texts: 原始文本列表
        translate_func: 翻译API调用函数
        source_lang: 源语言
        pivot_lang: 中转语言
        num_variants: 每个文本生成的变体数量

    Returns:
        增强后的文本变体列表
    """
    augmented = []
    for text in texts:
        for _ in range(num_variants):
            # 正向翻译: 源语言 → 中转语言
            pivot_text = translate_func(text, source_lang, pivot_lang)
            # 反向翻译: 中转语言 → 源语言
            back_text = translate_func(pivot_text, pivot_lang, source_lang)
            # 过滤与原文完全相同的结果
            if back_text.strip() != text.strip():
                augmented.append(back_text)
    return augmented


# 模拟翻译函数（实际使用时替换为真实翻译API）
def mock_translate(text: str, src: str, tgt: str) -> str:
    """模拟翻译 - 实际使用应接入翻译API"""
    variations = {
        "你好，今天天气怎么样？": "Hello, how is the weather today?",
        "Hello, how is the weather today?": "你好，今天天气如何？",
    }
    return variations.get(text, text)


def main():
    texts = ["你好，今天天气怎么样？"]
    augmented = back_translation_augment(texts, mock_translate)
    print("原始:", texts)
    print("增强:", augmented)
    print("OK")


if __name__ == "__main__":
    main()
