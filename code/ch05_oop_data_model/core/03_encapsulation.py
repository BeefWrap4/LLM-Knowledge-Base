# ---
# chapter: 5
# topic: Python 面向对象与数据模型
# topic_id: oop_data_model.encapsulation
# difficulty: ⭐⭐⭐
# tier: core
# deps: 无
# run: python 03_encapsulation.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../../../05_Python面向对象与数据模型.md
#
# Interview hooks:
# 1. Python 的封装和 Java/C++ 的封装有什么区别？
# 2. 计算属性（@property 不带 setter）和只读属性的区别？
# 3. 如何实现属性校验？除了 @property 还有别的方式吗？

"""
封装 —— 隐藏内部实现，暴露清晰接口
Python 通过命名约定实现封装（非强制）
"""


class Temperature:
    """
    温度类 —— 封装 Celsius 和 Fahrenheit 的转换逻辑
    """

    def __init__(self, celsius: float = 0):
        self._celsius = celsius  # 内部使用下划线前缀

    @property
    def celsius(self) -> float:
        """摄氏度 —— 只读属性"""
        return self._celsius

    @celsius.setter
    def celsius(self, value: float):
        """设置摄氏度，自动校验"""
        if value < -273.15:
            raise ValueError("温度不能低于绝对零度")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        """华氏度 —— 自动转换（计算属性）"""
        return self._celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float):
        """通过华氏度设置，反向转换"""
        self._celsius = (value - 32) * 5 / 9

    @property
    def kelvin(self) -> float:
        """开尔文"""
        return self._celsius + 273.15


# 使用 —— 封装隐藏了转换公式
t = Temperature(25)
print(f"{t.celsius}°C = {t.fahrenheit}°F = {t.kelvin}K")
t.fahrenheit = 98.6
print(f"{t.celsius}°C = {t.fahrenheit}°F")

if __name__ == "__main__":
    print("OK")
