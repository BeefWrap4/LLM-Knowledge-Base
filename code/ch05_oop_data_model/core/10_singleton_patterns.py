# ---
# chapter: 5
# topic: Python 面向对象与数据模型
# topic_id: oop_data_model.singleton_patterns
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: threading, functools
# run: python 10_singleton_patterns.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../../../05_Python面向对象与数据模型.md
#
# Interview hooks:
# 1. 手写一个线程安全的单例模式（要求说明双重检查锁的必要性）？
# 2. 三种单例实现（__new__ / 装饰器 / 元类）各自的优劣？
# 3. 装饰器实现单例后，类的 __name__ 为什么还是原来类名？@wraps 的作用？

"""
单例模式 —— 面试手撕代码超高频题

确保一个类只有一个实例，并提供一个全局访问点
"""

import threading

# ─────────────────────────────────────────────────────────────
# 方式1：__new__ 方法（最经典）
# ─────────────────────────────────────────────────────────────


class SingletonByNew:
    """
    通过 __new__ 实现单例

    原理：重写 __new__，在创建实例前检查是否已存在
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # 双重检查锁定
            with cls._lock:
                if cls._instance is None:  # 再次检查（防止并发创建）
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name=""):
        # ⚠️ 注意：__init__ 每次获取实例都会调用！
        if not hasattr(self, "_initialized"):
            self.name = name
            self._initialized = True


# 验证
s1 = SingletonByNew("first")
s2 = SingletonByNew("second")
print(f"同一实例? {s1 is s2}")  # True
print(f"name: {s1.name}")  # "first" — 第二次的初始化被忽略

# ─────────────────────────────────────────────────────────────
# 方式2：装饰器实现
# ─────────────────────────────────────────────────────────────

from functools import wraps


def singleton(cls):
    """
    单例装饰器 —— 最 Pythonic 的实现

    原理：装饰器返回一个包装函数，内部维护单一实例
    """
    instances = {}
    lock = threading.Lock()

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton
class Database:
    """数据库连接类 —— 单例"""

    def __init__(self, connection_string):
        self.connection_string = connection_string
        print(f"初始化数据库连接: {connection_string}")


db1 = Database("mysql://localhost")
db2 = Database("postgresql://remote")
print(f"同一实例? {db1 is db2}")  # True
print(f"连接字符串: {db2.connection_string}")  # "mysql://localhost"

# ─────────────────────────────────────────────────────────────
# 方式3：元类实现
# ─────────────────────────────────────────────────────────────


class SingletonMeta(type):
    """
    单例元类 —— 最底层的实现

    原理：控制类的创建过程，拦截 __call__ 方法
    """

    _instances = {}
    _locks = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            if cls not in cls._locks:
                cls._locks[cls] = threading.Lock()
            with cls._locks[cls]:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    """配置类 —— 单例"""

    def __init__(self):
        self.debug = False
        self.database_url = "sqlite:///default.db"


cfg1 = Config()
cfg2 = Config()
cfg1.debug = True
print(f"同一实例? {cfg1 is cfg2}")  # True
print(f"cfg2.debug = {cfg2.debug}")  # True — 共享状态

if __name__ == "__main__":
    print("OK")
