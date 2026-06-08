# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.8.2 防线4：Token 爆炸
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 13_token_limiter.py
# expected_runtime: <1s
# expected_output: 预算检查 + 输出截断结果
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.8.2-Agent-工程化安全五道防线
# Interview hooks:
#   1. 为什么 Token 爆炸是 Agent 特有的风险？(多步循环会累计)
#   2. 截断输出时如何在"截断位置"和"语义完整"之间权衡？(按段落/句子切分)
#   3. total_consumed 应该按请求估算还是按响应实际计费？两者差异怎么对账？


class TokenLimiter:
    """Token 限制器"""

    def __init__(self, max_output_tokens: int = 2000, max_total_tokens: int = 8000):
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.total_consumed = 0

    def check_budget(self, estimated_tokens: int) -> tuple[bool, dict]:
        """检查 Token 预算是否充足"""
        if self.total_consumed + estimated_tokens > self.max_total_tokens:
            return False, {
                "status": "budget_exceeded",
                "consumed": self.total_consumed,
                "budget": self.max_total_tokens,
                "action": "触发任务终止或摘要降级",
            }
        self.total_consumed += estimated_tokens
        return True, {"status": "ok", "remaining": self.max_total_tokens - self.total_consumed}

    def truncate_output(self, text: str, max_length: int = None) -> str:
        """截断输出"""
        max_len = max_length or self.max_output_tokens
        if len(text) <= max_len:
            return text
        return text[:max_len] + "\n...[输出已截断]"


def main():
    limiter = TokenLimiter(max_output_tokens=100, max_total_tokens=300)

    print("=== 逐步消耗 Token 预算 ===")
    for est in [50, 80, 100, 90]:
        ok, info = limiter.check_budget(est)
        print(f"  请求 {est} tokens -> ok={ok} | {info}")

    long_output = "这是一段超长输出 " * 100  # 约 1000 字符
    truncated = limiter.truncate_output(long_output, max_length=50)
    print("\n=== 输出截断 ===")
    print(f"  原长度: {len(long_output)}")
    print(f"  截断后: {truncated}")
    print("\nOK")


if __name__ == "__main__":
    main()
