# ---
# chapter: 22
# topic: 大模型数据工程
# section: 22.2.4 毒性过滤与 PII 去除
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: stdlib only
# run: python 03_pii_scrubber.py
# expected_runtime: <2s
# expected_output: 脱敏后的字符串和检测到的 PII 列表
# ---
# See: ../tutorial/22_大模型数据工程.md
#
# Interview hooks:
#   1. 大规模 PII 检测面临哪些核心挑战？正则 + NER 的混合方案有何优劣？
#   2. 为什么"宁可多脱敏（假阳性）不可漏检（假阴性）"是 PII 处理的核心原则？
#   3. Llama Guard 等安全分类器在 PII 检测中扮演什么角色？

import re


class PIIScrubber:
    """PII 检测与脱敏"""

    # 常见 PII 正则模式
    PATTERNS = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "IPV4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        "PHONE_CN": re.compile(r"\b1[3-9]\d{9}\b"),
        "PHONE_US": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "URL": re.compile(r'https?://[^\s<>"{}|\[\]]+'),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[ -]?){4}\b"),
    }

    @classmethod
    def scrub_text(cls, text: str, use_ner: bool = False) -> tuple[str, list[dict]]:
        """对文本进行 PII 脱敏"""
        findings = []
        scrubbed = text

        for pii_type, pattern in cls.PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "type": pii_type,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
                scrubbed = scrubbed.replace(match.group(), f"[{pii_type}]")

        return scrubbed, findings

    @classmethod
    def check_pii_free(cls, text: str) -> bool:
        """检查文本是否 PII-free"""
        for pattern in cls.PATTERNS.values():
            if pattern.search(text):
                return False
        return True


def main():
    # 使用示例
    sample = "联系邮箱 john.doe@example.com，电话 13800138000，IP地址 192.168.1.1"
    cleaned, found = PIIScrubber.scrub_text(sample)
    print(f"脱敏后: {cleaned}")
    print(f"检测到的 PII: {found}")
    # 脱敏后: 联系邮箱 [EMAIL]，电话 [PHONE_CN]，IP地址 [IPV4]
    print(f"是否 PII-free: {PIIScrubber.check_pii_free(sample)}")
    print(f"脱敏后是否 PII-free: {PIIScrubber.check_pii_free(cleaned)}")


if __name__ == "__main__":
    main()
