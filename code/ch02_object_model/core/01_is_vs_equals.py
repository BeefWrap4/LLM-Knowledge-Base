# ---
# chapter: 2
# topic: Python 对象模型与可变性
# topic_id: object_model.is_vs_equals
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: stdlib
# run: python 01_is_vs_equals.py
# expected_runtime: <1s
# ---
#
# See: ../../../02_Python对象模型与可变性.md
# Interview hooks:
#   - "is 和 == 区别?"           →  is: id() 身份; ==: __eq__() 值相等
#   - "小整数缓存范围?"          →  CPython 常见为 -5 ~ 256，但属于实现细节
#   - "字符串驻留 (interning)?"  →  编译期常量自动驻留; 运行时拼接不驻留

def main() -> None:
    """Demonstrate identity versus value equality without relying on interning."""
    # is 比较身份；== 比较值（调用 __eq__）。
    cached_a = 256
    cached_b = 256
    print(f"256 is 256: {cached_a is cached_b}")
    print(f"256 == 256: {cached_a == cached_b}")

    # 不要依赖常量折叠推断缓存范围；运行时构造才能稳定展示不同对象。
    dynamic_a = int("257")
    dynamic_b = int("257")
    print(f"runtime 257 is runtime 257: {dynamic_a is dynamic_b}")
    print(f"runtime 257 == runtime 257: {dynamic_a == dynamic_b}")

    s1 = "hello"
    s2 = "".join(["hel", "lo"])
    print(f"'hello' is runtime string: {s1 is s2}")
    print(f"'hello' == runtime string: {s1 == s2}")

    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    print(f"[1,2,3] is [1,2,3]: {list1 is list2}")
    print(f"[1,2,3] == [1,2,3]: {list1 == list2}")

    t1 = tuple([1, 2, 3])
    t2 = tuple([1, 2, 3])
    print(f"(1,2,3) is runtime tuple: {t1 is t2}")
    print(f"(1,2,3) == runtime tuple: {t1 == t2}")
    print("OK")


if __name__ == "__main__":
    main()
