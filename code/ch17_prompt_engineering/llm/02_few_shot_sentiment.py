# ---
# chapter: 17
# topic: Prompt Engineering
# topic_id: prompt_engineering.few_shot_sentiment
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: 无外部依赖
# run: python 02_few_shot_sentiment.py
# expected_runtime: <1s
# expected_output: 打印 Few-shot 情感分类 Prompt
# ---
# See: ../../../17_Prompt_Engineering.md
# Interview hooks:
# - Few-shot 示例数量多少最佳？为何 5+ 会出现边际递减？
# - 示例格式不统一会带来什么后果？
# - 为何"边界案例"对示例选择尤为关键？

# Few-shot Prompting 示例：情感分类
few_shot_prompt = """
请根据以下示例，判断每条评论的情感倾向（正面/负面/中性）。

示例1：
评论："这款手机拍照效果太棒了，夜景模式非常清晰！"
情感：正面

示例2：
评论："物流慢得要死，等了一周才到。"
情感：负面

示例3：
评论："产品一般，和价格匹配。"
情感：中性

---

待分类评论："客服态度很好，但产品质量有待提升。"
情感："""

# 预期输出：中性（或 混合，取决于模型理解）


if __name__ == "__main__":
    print(few_shot_prompt)
    print("\n[预期输出] 中性（或 混合，取决于模型理解）")
    print("OK")
