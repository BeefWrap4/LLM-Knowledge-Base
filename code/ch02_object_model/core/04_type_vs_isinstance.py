# ---
# chapter: 2
# topic: Python 对象模型与可变性
# topic_id: object_model.type_vs_isinstance
# difficulty: ⭐⭐
# tier: core
# deps: 无
# run: python 04_type_vs_isinstance.py
# expected_runtime: <1s
# expected_output: 演示 type() 与 isinstance() 在继承场景下的差异
# ---
# See: ../../../02_Python对象模型与可变性.md
# Interview hooks:
#   1. type() 和 isinstance() 的核心区别是什么?
#   2. 实际开发中应该优先使用哪个?为什么?
#   3. 什么场景下必须用 type() 而不是 isinstance()?

"""
type() vs isinstance() —— 面试高频考点

核心区别:isinstance() 考虑继承关系(鸭子类型),type() 不考虑
"""


class Animal:
    pass


class Dog(Animal):
    pass


dog = Dog()

# type() — 返回精确类型
print(type(dog))  # <class '__main__.Dog'>
print(type(dog) == Dog)  # True
print(type(dog) == Animal)  # False — Animal 不是 Dog 的精确类型

# isinstance() — 考虑继承链
print(isinstance(dog, Dog))  # True
print(isinstance(dog, Animal))  # True — Dog 继承自 Animal

# ─────────────────────────────────────────────────────────────
# 🎯 面试陷阱:type() 判断子类
# ─────────────────────────────────────────────────────────────


class MyList(list):
    """自定义列表类"""

    pass


my_list = MyList([1, 2, 3])


# 错误写法
def process_list_bad(data):
    if type(data) == list:  # ❌ 不会匹配 MyList 实例
        print("是 list")
    else:
        print("不是 list")


# 正确写法
def process_list_good(data):
    if isinstance(data, list):  # ✅ 匹配 list 及其所有子类
        print("是 list 或其子类")
    else:
        print("不是 list")


process_list_bad(my_list)  # "不是 list" — 错误!
process_list_good(my_list)  # "是 list 或其子类" — 正确!

# ─────────────────────────────────────────────────────────────
# 多类型判断
# ─────────────────────────────────────────────────────────────


def flexible_add(a, b):
    """支持数字或字符串相加"""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b
    elif isinstance(a, str) and isinstance(b, str):
        return a + b
    else:
        raise TypeError("只支持数字或字符串")


print(flexible_add(3, 4))  # 7
print(flexible_add("a", "b"))  # "ab"
# flexible_add(3, "b")            # TypeError

# ─────────────────────────────────────────────────────────────
# type() 的应用场景:精确类型判断
# ─────────────────────────────────────────────────────────────


def strict_type_check(obj):
    """需要精确类型匹配的场景(如反序列化)"""
    if type(obj) is dict:  # 必须是 dict,不能是子类
        print("原生 dict")
    elif type(obj) is list:
        print("原生 list")
    else:
        print(f"其他类型: {type(obj).__name__}")


strict_type_check({})  # "原生 dict"
strict_type_check(MyList())  # "原生 list" — 等等...
# 实际上 MyList() 是 list 子类,type(MyList()) 是 MyList,不是 list
# 所以这里会输出 "其他类型: MyList"


# 正确的精确判断
def exact_type_check(obj, expected_type):
    """精确判断 obj 的类型就是 expected_type(非子类)"""
    return type(obj) is expected_type


print(exact_type_check({}, dict))  # True
print(exact_type_check(MyList(), list))  # False — MyList 不是 list

if __name__ == "__main__":
    print("OK")
