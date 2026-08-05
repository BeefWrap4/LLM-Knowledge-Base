# ---
# chapter: 5
# topic: Python 面向对象与数据模型
# topic_id: oop_data_model.singleton
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: stdlib
# run: python 01_singleton.py
# expected_runtime: <1s
# ---
#
# See: ../../../05_Python面向对象与数据模型.md
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


def main() -> None:
    s1 = SingletonNew("first")
    s2 = SingletonNew("second")
    print(f"__new__ 方式: s1 is s2 = {s1 is s2}")
    print(f"  s1.name = {s1.name}, s2.name = {s2.name}  # __init__ 会再次执行")

    d1 = SingletonDec(42)
    d2 = SingletonDec(100)
    print(f"\n装饰器方式: d1 is d2 = {d1 is d2}")
    print(f"  d1.value = {d1.value}, d2.value = {d2.value}  # 共享第一个实例")

    m1 = SingletonMeta_("alpha")
    m2 = SingletonMeta_("beta")
    print(f"\n元类方式: m1 is m2 = {m1 is m2}")
    print(f"  m1.name = {m1.name}")

    assert s1 is s2, "__new__ 单例失败"
    assert d1 is d2, "装饰器单例失败"
    assert m1 is m2, "元类单例失败"
    print("OK")


if __name__ == "__main__":
    main()
