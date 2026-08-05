# ---
# chapter: 5
# topic: Python 面向对象与数据模型
# topic_id: oop_data_model.metaclass
# difficulty: ⭐⭐
# tier: core
# deps: 无
# run: python 14_metaclass.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../../../05_Python面向对象与数据模型.md
#
# Interview hooks:
# 1. 什么是元类？type 和普通自定义类的关系？
# 2. 元类 __new__ 和 __init__ 的调用顺序和参数差异？
# 3. 元类的实际应用场景（Django ORM、SQLAlchemy、单例）？

"""
元类（Metaclass）— 类的类

 everything in Python is an object, including classes.
 Classes are instances of type (or its subclass).

 type(name, bases, namespace) → 创建新类

┌─────────────────────────────────────────────────────────────┐
│                     元类层级                                 │
│                                                             │
│   type ──是──► type 的元类                                  │
│   │                                                         │
│   │ 是 MyClass 的元类                                       │
│   ▼                                                         │
│   MyClass ──是──► MyClass() 的类                            │
│   │                                                         │
│   │ 是 obj 的类                                             │
│   ▼                                                         │
│   obj = MyClass()                                           │
│                                                             │
│   isinstance(obj, MyClass)    → True                        │
│   isinstance(MyClass, type)   → True                        │
│   isinstance(type, type)      → True（type 是自己的实例）     │
└─────────────────────────────────────────────────────────────┘
"""

# ─────────────────────────────────────────────────────────────
# 用 type 动态创建类
# ─────────────────────────────────────────────────────────────


def say_hello(self):
    return f"Hello, I'm {self.name}"


# type(name, bases, namespace)
DynamicClass = type(
    "DynamicClass",  # 类名
    (object,),  # 基类元组
    {  # 属性字典
        "__init__": lambda self, name: setattr(self, "name", name),
        "say_hello": say_hello,
    },
)

obj = DynamicClass("Dynamic")
print(obj.say_hello())  # "Hello, I'm Dynamic"

# ─────────────────────────────────────────────────────────────
# 自定义元类
# ─────────────────────────────────────────────────────────────


class ValidateMeta(type):
    """
    元类 —— 在类创建时自动验证属性
    """

    def __new__(mcs, name, bases, namespace):
        """创建类对象之前 —— 可修改 namespace"""
        print(f"创建类: {name}")

        # 自动添加 __slots__（节省内存）
        # 排除: dunder / 保留名 / 有类级默认值的属性 (后两者会与 slot 冲突)
        if "__slots__" not in namespace and bases == ():
            reserved = {"name", "doc", "qualname", "module", "dict", "weakref"}
            attrs = []
            for k, v in namespace.items():
                if k.startswith("__") or k in reserved:
                    continue
                if callable(v):
                    continue  # 方法不放 slot
                # 任何带类级默认值的属性都不能进 slot
                attrs.append(k)
            if attrs:
                # 但如果 attrs 中含默认值的项, 与 class var 冲突, 需剔除
                filtered = []
                for k in attrs:
                    v = namespace[k]
                    # 严格标准: 若有非方法且非类级哨兵值, 跳过
                    if isinstance(v, type(lambda: 0)) and v.__class__.__name__ == "function":
                        filtered.append(k)
                    elif hasattr(v, "__class__") and v.__class__.__module__ == "builtins":
                        # 含类级默认值的内置类型, 跳过避免冲突
                        continue
                    else:
                        filtered.append(k)
                if filtered:
                    namespace["__slots__"] = filtered

        # 强制方法命名规范
        for attr_name in namespace:
            if callable(namespace[attr_name]) and attr_name.startswith("Get"):
                raise TypeError(f"方法名 {attr_name} 不符合规范，应使用小写+下划线")

        cls = super().__new__(mcs, name, bases, namespace)
        return cls

    def __init__(cls, name, bases, namespace):
        """类对象创建后 —— 可添加类属性、注册类等"""
        super().__init__(name, bases, namespace)
        cls.class_timestamp = "2025-01-01"


class Product(metaclass=ValidateMeta):
    """使用自定义元类的类"""

    name = ""
    price = 0.0

    def __init__(self, name, price):
        self.name = name
        self.price = price


# Product 的创建过程中 ValidateMeta.__new__ 和 __init__ 被调用
print(f"Product.class_timestamp: {Product.class_timestamp}")

# ─────────────────────────────────────────────────────────────
# 元类实现单例（回顾）
# ─────────────────────────────────────────────────────────────


class SingletonMeta(type):
    """单例元类 —— 控制实例创建"""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AppConfig(metaclass=SingletonMeta):
    def __init__(self):
        self.debug = False


c1 = AppConfig()
c2 = AppConfig()
print(f"元类单例: {c1 is c2}")  # True

# ─────────────────────────────────────────────────────────────
# 元类的应用场景（面试常问）
# ─────────────────────────────────────────────────────────────

"""
元类的主要应用场景：

1. ORM 框架（Django ORM、SQLAlchemy）
   - 自动将类属性映射为数据库字段
   - 自动生成查询方法

2. API 框架（FastAPI、DRF）
   - 自动从类定义生成 API 路由
   - 自动序列化/反序列化

3. 注册模式
   - 类创建时自动注册到某个注册表

4. 代码生成/转换
   - 自动添加方法、修改属性
   - 接口校验

5. 单例模式（如上所示）
"""

if __name__ == "__main__":
    print("OK")
