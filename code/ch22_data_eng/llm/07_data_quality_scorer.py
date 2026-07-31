# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.5.2 自动化质量评分
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: numpy
# run: python 07_data_quality_scorer.py
# expected_runtime: <3s
# expected_output: 总分 + 详细分数字典
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. SFT 数据质量的多维评分体系如何设计？各维度的权重如何权衡？
#   2. type_token_ratio (TTR) 为什么能反映文本多样性？它有什么局限？
#   3. 自动化评分和 LLM-as-Judge 各有什么优劣？如何组合使用？


import numpy as np


class DataQualityScorer:
    """SFT 数据质量自动评分器"""

    @staticmethod
    def compute_text_stats(text: str) -> dict:
        """计算文本统计特征"""
        words = text.split()
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
            "avg_sentence_length": float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0,
            "type_token_ratio": len(set(words)) / len(words) if words else 0.0,
            "char_diversity": len(set(text)) / len(text) if text else 0.0,
        }

    @staticmethod
    def check_format_compliance(text: str, expected_format: str = "general") -> dict:
        """检查格式合规性"""
        checks = {}

        if expected_format == "json":
            import json as json_module

            try:
                json_module.loads(text)
                checks["valid_json"] = True
            except json_module.JSONDecodeError:
                checks["valid_json"] = False

        # 检查是否有截断
        checks["likely_truncated"] = text.rstrip().endswith((",", "、", "and", "...", "等等"))

        # 检查代码块配对
        checks["code_blocks_balanced"] = text.count("```") % 2 == 0

        return checks

    def score_instruction_data(self, item: dict) -> dict:
        """
        对单条指令数据进行综合评分

        Args:
            item: {"instruction": str, "input": str, "output": str}
        """
        inst = item.get("instruction", "")
        output = item.get("output", "")
        full_text = inst + " " + output

        stats = self.compute_text_stats(full_text)
        fmt = self.check_format_compliance(output)

        # 评分逻辑
        scores = {
            "length_score": min(stats["word_count"] / 100, 1.0) * 100,
            "diversity_score": stats["type_token_ratio"] * 100,
            "format_score": 100 if all(fmt.values()) else 60,
            "not_truncated": 100 if not fmt.get("likely_truncated") else 0,
        }

        # 加权总分
        weights = {
            "length_score": 0.2,
            "diversity_score": 0.3,
            "format_score": 0.3,
            "not_truncated": 0.2,
        }
        total = sum(scores[k] * weights[k] for k in weights)

        return {
            "total_score": round(total, 1),
            "detail_scores": scores,
            "stats": stats,
            "format_checks": fmt,
        }


def main():
    # 示例
    scorer = DataQualityScorer()
    sample = {
        "instruction": "用 Python 实现二分查找",
        "input": "",
        "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    }
    result = scorer.score_instruction_data(sample)
    print(f"总分: {result['total_score']}")
    print(f"详细: {result['detail_scores']}")
    print("OK")


if __name__ == "__main__":
    main()
