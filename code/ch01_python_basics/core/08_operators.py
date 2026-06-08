# ---
# chapter: 1
# topic: 运算符优先级与特殊运算符
# section: 1.2.2
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 08_operators.py
# expected_runtime: <1s
# expected_output: 运算符示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 306-360)
# Interview hooks:
#   1. is 和 == 的本质区别?为什么 None 必须用 is?
#   2. Python 中 falsy 值有哪些?
#   3. 海象运算符 := 的典型应用场景?
"""
运算符优先级与特殊运算符
"""

# ─────────────────────────────────────────────────────────────
# 身份运算符 is / is not（面试高频考点）
# ─────────────────────────────────────────────────────────────

# is 比较内存地址，== 比较值
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True — 值相等
print(a is b)  # False — 内存地址不同


# None 比较必须用 is
def check_none(x):
    """判断 None 的正确方式"""
    if x is None:  # ✅ 正确
        return "是 None"
    # if x == None:    # ❌ 不规范
    return "不是 None"


print(check_none(None))
print(check_none(0))

# ─────────────────────────────────────────────────────────────
# 短路求值（and / or）
# ─────────────────────────────────────────────────────────────


def get_fallback(value, default):
    """利用 or 短路求值提供默认值"""
    return value or default  # value 为 falsy 时返回 default


# falsy 值：0, 0.0, "", [], {}, set(), None, False
print(get_fallback("", "default"))  # "default"
print(get_fallback("hello", "def"))  # "hello"
print(get_fallback(0, 42))  # 42


# 安全获取嵌套字典值（Python 3.8+ 海象运算符 :=）
def get_nested(data: dict, key1: str, key2: str):
    """使用海象运算符简化嵌套获取"""
    if (inner := data.get(key1)) is not None:
        return inner.get(key2)
    return None


sample = {"user": {"name": "Alice"}}
print(f"嵌套获取: {get_nested(sample, 'user', 'name')}")

# 海象运算符在 while 循环中的应用
numbers = [3, 1, 4, 1, 5, 9]
it = iter(numbers)
# 传统写法需要两次 next() 调用
# while True:
#     n = next(it, None)
#     if n is None: break
#     print(n)
# 海象运算符版本更简洁
# while (n := next(it, None)) is not None:
#     print(n)

if __name__ == "__main__":
    print("OK")
