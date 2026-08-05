# ---
# chapter: 3
# topic: Python 函数、作用域与装饰器
# topic_id: iteration_functional.closure_basics
# difficulty: ⭐⭐⭐
# tier: core
# deps: 无
# run: python 01_closure_basics.py
# expected_runtime: <1s
# expected_output: 闭包基础示例 — 输出 add_10(5)=15, add_20(5)=25
# ---
# See: ../../../03_Python函数作用域与装饰器.md
# Interview hooks:
#   1. 闭包的三要素是什么？
#   2. 闭包与普通函数有什么本质区别？
#   3. __closure__ 属性的 cell_contents 存的是什么？

"""
闭包（Closure）— 面试高频考点

定义：闭包 = 嵌套函数 + 引用外部变量 + 返回嵌套函数

闭包的三要素：
1. 必须有一个嵌套函数
2. 嵌套函数必须引用外部函数中的变量
3. 外部函数必须返回嵌套函数

闭包的本质：函数记住并访问它被创建时的词法环境
"""

# 闭包基础示例


def outer(x):  # 外部函数
    def inner(y):  # 嵌套函数 —— 闭包
        return x + y  # inner 引用了外部变量 x

    return inner  # 返回嵌套函数（不是调用！）


# 创建两个不同的闭包
add_10 = outer(10)  # add_10 是一个闭包，记住了 x=10
add_20 = outer(20)  # add_20 是一个闭包，记住了 x=20

print(add_10(5))  # 15 — 10 + 5
print(add_20(5))  # 25 — 20 + 5

# 验证闭包记住了外部变量
print(add_10.__closure__[0].cell_contents)  # 10
print(add_20.__closure__[0].cell_contents)  # 20


if __name__ == "__main__":
    print("OK")
