# ---
# chapter: 3
# topic: 单例模式三种实现 (__new__, 装饰器, 元类)
# section: 3.4
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: stdlib
# run: python 01_singleton.py
# expected_runtime: <1s
# ---
#
# See: ../tutorial/Ch03_Python面向对象编程.md §3.4
# Interview hooks:
#   - "单例模式三种实现?"  →  __new__ / 装饰器 / 元类
#   - "为什么需要单例?"    →  配置类、连接池、日志器
#   - "线程安全的单例?"    →  加 threading.Lock

# ─────────────────────────────────────────────
# 方式1: __new__ 重写
# ─────────────────────────────────────────────


class SingletonNew:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name):
        self.name = name  # 注意: 多次实例化会覆盖 name


s1 = SingletonNew("first")
s2 = SingletonNew("second")
print(f"__new__ 方式: s1 is s2 = {s1 is s2}")
print(f"  s1.name = {s1.name}, s2.name = {s2.name}  # 都是 'second' 因为共享 __init__")


# ─────────────────────────────────────────────
# 方式2: 装饰器
# ─────────────────────────────────────────────


def singleton_decorator(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton_decorator
class SingletonDec:
    def __init__(self, value):
        self.value = value


d1 = SingletonDec(42)
d2 = SingletonDec(100)
print(f"\n装饰器方式: d1 is d2 = {d1 is d2}")
print(f"  d1.value = {d1.value}, d2.value = {d2.value}  # 共享第一个实例")


# ─────────────────────────────────────────────
# 方式3: 元类 (最强大但复杂)
# ─────────────────────────────────────────────


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in SingletonMeta._instances:
            SingletonMeta._instances[cls] = super().__call__(*args, **kwargs)
        return SingletonMeta._instances[cls]


class SingletonMeta_(metaclass=SingletonMeta):
    def __init__(self, name):
        self.name = name


m1 = SingletonMeta_("alpha")
m2 = SingletonMeta_("beta")
print(f"\n元类方式: m1 is m2 = {m1 is m2}")
print(f"  m1.name = {m1.name}")


# ─────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────
assert s1 is s2, "__new__ 单例失败"
assert d1 is d2, "装饰器单例失败"
assert m1 is m2, "元类单例失败"

print("\nOK")

print("OK")
