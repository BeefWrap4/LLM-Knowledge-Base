# ---
# chapter: 39
# topic: AI 隐私、伦理与治理
# topic_id: safety.data_anonymizer
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: none
# run: python 08_data_anonymizer.py
# expected_runtime: <1s
# expected_output: PII脱敏演示 + k-匿名性检查 + "OK"
# ---
# See: ../../../39_AI隐私伦理与治理.md
# Interview hooks:
#   1. 匿名化（不可逆）与去标识化（可逆）的区别是什么？
#   2. k-匿名性如何防止重标识攻击？k值的选择标准？
#   3. 在大模型训练场景下，PII检测与脱敏面临哪些特殊挑战？
"""
数据匿名化处理示例

面试要点：
1. 区分匿名化（不可逆）vs 去标识化（可逆）
2. k-匿名性（k-anonymity）概念
3. 大模型场景下的特殊考虑
"""

import hashlib
import re


class DataAnonymizer:
    """数据匿名化处理器"""

    # PII（个人身份信息）模式
    PII_PATTERNS = {
        "email": r"[\w\.-]+@[\w\.-]+\.\w+",
        "phone_cn": r"1[3-9]\d{9}",
        "id_card": r"\d{17}[\dXx]",
        "ip": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    }

    def __init__(self, salt: str = ""):
        """
        Args:
            salt: 哈希盐值，确保不可逆
        """
        self.salt = salt
        self.replacement_map = {}  # 储存替换映射（生产环境中仅存内存）

    def anonymize_text(self, text: str) -> str:
        """对文本中的PII进行匿名化"""
        cleaned = text

        # 1. 邮箱脱敏
        for match in re.finditer(self.PII_PATTERNS["email"], cleaned):
            email = match.group()
            replacement = f"[EMAIL_{self._hash(email)[:8]}]"
            self.replacement_map[email] = replacement
            cleaned = cleaned.replace(email, replacement)

        # 2. 手机号脱敏
        for match in re.finditer(self.PII_PATTERNS["phone_cn"], cleaned):
            phone = match.group()
            # 保留前3后4，中间替换
            replacement = phone[:3] + "****" + phone[-4:]
            cleaned = cleaned.replace(phone, replacement)

        # 3. 身份证号脱敏
        for match in re.finditer(self.PII_PATTERNS["id_card"], cleaned):
            id_card = match.group()
            replacement = id_card[:6] + "********" + id_card[-4:]
            cleaned = cleaned.replace(id_card, replacement)

        return cleaned

    def _hash(self, value: str) -> str:
        """不可逆哈希"""
        return hashlib.sha256((value + self.salt).encode()).hexdigest()

    def check_k_anonymity(self, dataset: list[list], k: int = 5) -> dict:
        """检查k-匿名性

        k-匿名性：每个准标识符组合在数据集中至少出现k次。
        """
        from collections import Counter

        # 将每个记录的准标识符转为元组并计数
        records = [tuple(record) for record in dataset]
        counts = Counter(records)

        violations = {record: count for record, count in counts.items() if count < k}

        return {
            "total_records": len(dataset),
            "unique_combinations": len(counts),
            "k_target": k,
            "violations": len(violations),
            "k_anonymous": len(violations) == 0,
            "min_group_size": min(counts.values()),
            "max_group_size": max(counts.values()),
        }


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("=== 数据匿名化演示 ===")

    anonymizer = DataAnonymizer(salt="my-secret-salt-2026")

    # 1. PII脱敏演示
    test_text = "我的邮箱是user@example.com，手机号是13812345678，身份证110101199001011234"
    cleaned = anonymizer.anonymize_text(test_text)
    print(f"\n原始文本: {test_text}")
    print(f"脱敏后:   {cleaned}")

    # 2. k-匿名性检查演示
    print("\n=== k-匿名性检查 ===")
    # 数据集：每条记录包含[年龄区间, 邮编, 性别]等准标识符
    dataset = [
        ["20-30", "100000", "M"],
        ["20-30", "100000", "M"],  # 重复组合
        ["20-30", "100000", "F"],
        ["40-50", "200000", "F"],
        ["40-50", "200000", "F"],
    ]
    result = anonymizer.check_k_anonymity(dataset, k=3)
    print(f"  k值: {result['k_target']}")
    print(f"  总记录数: {result['total_records']}")
    print(f"  唯一组合数: {result['unique_combinations']}")
    print(f"  违规组合数: {result['violations']}")
    print(f"  k-匿名合规: {'✅ 是' if result['k_anonymous'] else '❌ 否'}")
    print(f"  最小/最大组规模: {result['min_group_size']}/{result['max_group_size']}")
    print("OK")
