# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.2.3 质量过滤 - 多维度文本质量过滤
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: fasttext (optional)
# run: python 02_text_quality_filter.py
# expected_runtime: <5s
# expected_output: good_text / bad_text 的过滤结果 dict
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. 质量过滤中"先快后慢"的分层原则是什么？为什么要这样设计？
#   2. 启发式规则 vs 困惑度过滤 vs 分类器过滤各有什么优缺点？
#   3. FineWeb-Edu 分类器的工作原理是什么？它为什么能成为 2024-2026 的主流方案？

import re
from collections import Counter

# Mock fasttext 以避免在无 fasttext 环境下报错
try:
    import fasttext  # type: ignore
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False

    class _MockFastText:
        """Mock 实现，模拟 fasttext 模型接口（仅用于演示）"""

        def predict(self, text: str):
            return (("__label__en",), (0.99,))

    fasttext = type("fasttext", (), {"load_model": staticmethod(lambda *a, **kw: _MockFastText())})


class TextQualityFilter:
    """多维度文本质量过滤器"""

    def __init__(self, lang_model_path: str = "lid.176.bin"):
        # fastText 语言检测模型（需预先下载）
        try:
            self.lang_model = fasttext.load_model(lang_model_path)
        except Exception:
            print("⚠️  fastText 模型未找到，语言检测将被跳过")
            self.lang_model = None

    @staticmethod
    def heuristic_filter(text: str) -> tuple[bool, str]:
        """启发式规则过滤"""
        text_stripped = text.strip()

        # 1. 长度过滤
        if len(text_stripped) < 50:
            return False, "too_short"
        if len(text_stripped) > 100000:
            return False, "too_long"

        # 2. 特殊字符比例
        special_ratio = sum(1 for c in text_stripped
                            if not c.isalnum() and not c.isspace()) / max(len(text_stripped), 1)
        if special_ratio > 0.3:
            return False, f"high_special_char_ratio_{special_ratio:.2f}"

        # 3. 大写比例（适用于英文）
        alpha_chars = [c for c in text_stripped if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.5 and len(text_stripped) > 200:
                return False, f"high_upper_ratio_{upper_ratio:.2f}"

        # 4. 行重复检测
        lines = text_stripped.split('\n')
        if len(lines) > 5:
            line_counts = Counter(lines)
            most_common_ratio = line_counts.most_common(1)[0][1] / len(lines)
            if most_common_ratio > 0.5:
                return False, f"high_line_repetition_{most_common_ratio:.2f}"

        # 5. 停用词比例（以英文为例）
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                     'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        words = text_stripped.lower().split()
        if len(words) > 20:
            stopword_ratio = sum(1 for w in words if w in stopwords) / len(words)
            if stopword_ratio < 0.05:
                return False, f"low_stopword_ratio_{stopword_ratio:.3f}"

        return True, "pass"

    def language_filter(self, text: str, target_lang: str = "__label__en") -> tuple[bool, str]:
        """语言检测过滤"""
        if self.lang_model is None:
            return True, "lang_skipped"
        text_clean = re.sub(r'\s+', ' ', text.strip())[:500]  # 取前500字符
        pred = self.lang_model.predict(text_clean)
        detected_lang = pred[0][0]
        confidence = pred[1][0]
        return detected_lang == target_lang and confidence > 0.7, f"{detected_lang}_{confidence:.2f}"

    def full_filter(self, text: str, target_lang: str = "__label__en") -> dict:
        """综合过滤"""
        h_pass, h_reason = self.heuristic_filter(text)
        l_pass, l_reason = self.language_filter(text, target_lang)

        return {
            "passed": h_pass and l_pass,
            "heuristic_pass": h_pass,
            "heuristic_reason": h_reason,
            "language_pass": l_pass,
            "language_reason": l_reason,
        }


def main():
    # 使用示例
    filter_obj = TextQualityFilter()

    good_text = ("Machine learning is a field of inquiry devoted to understanding "
                 "and building methods that learn from data. It is seen as a part "
                 "of artificial intelligence. Machine learning algorithms build a "
                 "model based on sample data, known as training data, in order to "
                 "make predictions or decisions without being explicitly programmed "
                 "to do so.")
    bad_text = "asdf jkl; @#$%^&* !!! " * 10  # 低质量文本

    print("Good text:", filter_obj.full_filter(good_text))
    print("Bad text:", filter_obj.full_filter(bad_text))


if __name__ == "__main__":
    main()
