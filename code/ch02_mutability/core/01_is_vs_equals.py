# ---
# chapter: 2
# topic: is vs == 区别 + 小整数缓存
# section: 2.1.3
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: stdlib
# run: python 01_is_vs_equals.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/Ch02_Python核心面试专题_可变性与拷贝.md §2.1.3
# Interview hooks:
#   - "is 和 == 区别?"           →  is: id() 身份; ==: __eq__() 值相等
#   - "小整数缓存范围?"          →  -5 ~ 256 (sys.getswitchinterval 范围)
#   - "字符串驻留 (interning)?"  →  编译期常量自动驻留; 运行时拼接不驻留

# is 比较身份 (内存地址)
# == 比较值 (调用 __eq__)

# 1. 小整数缓存
a = 256
b = 256
print(f"256 is 256: {a is b}")  # True — 缓存
print(f"256 == 256: {a == b}")  # True

a = 257
b = 257
print(f"257 is 257: {a is b}")  # False — 超出缓存
print(f"257 == 257: {a == b}")  # True

# 2. 字符串驻留 (interning)
s1 = "hello"  # 字面量, 自动驻留
s2 = "hello"
print(f"'hello' is 'hello': {s1 is s2}")  # True
print(f"'hello' == 'hello': {s1 == s2}")  # True

# 运行时拼接 — 不驻留
s3 = "hel" + "lo"
print(f"拼接 is: {s1 is s3}")  # False
print(f"拼接 ==: {s1 == s3}")  # True

# 4. 列表 is vs ==
list1 = [1, 2, 3]
list2 = [1, 2, 3]
print(f"\n[1,2,3] is [1,2,3]: {list1 is list2}")  # False — 不同对象
print(f"[1,2,3] == [1,2,3]: {list1 == list2}")  # True — 值相等
print(f"id(list1)={id(list1)}, id(list2)={id(list2)}")

# 5. 重要: 不可变对象 is 比 == 更快

# 字符串 is 比 == 稍快 (因为不需要调用 __eq__)
s1 = "python"
s2 = "python"
print("\nstr is 速度更快 (避免 __eq__ 调用)")

# tuple
t1 = (1, 2, 3)
t2 = (1, 2, 3)
print(f"\n(1,2,3) is (1,2,3): {t1 is t2}")  # True (Py 3.10+ tuple interning)
print(f"(1,2,3) == (1,2,3): {t1 == t2}")

print("\nOK")

print("OK")
