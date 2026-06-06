# ---
# chapter: 3
# topic: 魔术方法大全
# section: 3.5 魔术方法大全
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: functools
# run: python 11_magic_methods.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/03_Python面向对象编程.md#3.5-魔术方法大全
#
# Interview hooks:
# 1. __repr__ 和 __str__ 的区别？什么时候用哪个？
# 2. 定义了 __eq__ 之后为什么通常还要重定义 __hash__？
# 3. __add__ 和 __radd__ 的关系？什么场景下必须实现 __radd__？

"""
魔术方法（Magic/Dunder Methods）— 以双下划线开头和结尾的特殊方法

分类：
1. 生命周期方法
2. 字符串表示方法
3. 比较方法
4. 算术运算方法
5. 容器类型方法
6. 可调用对象
7. 上下文管理器
8. 属性访问
"""

from functools import total_ordering

# ─────────────────────────────────────────────────────────────
# 完整魔术方法示例类
# ─────────────────────────────────────────────────────────────

@total_ordering   # 自动生成剩余比较方法
class Vector2D:
    """
    二维向量 —— 演示各类魔术方法
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    # ── 1. 字符串表示 ──
    def __repr__(self):
        """面向开发者的表示 —— eval(repr(obj)) 应能重建对象"""
        return f"Vector2D({self.x!r}, {self.y!r})"

    def __str__(self):
        """面向用户的表示 —— print() 调用"""
        return f"({self.x}, {self.y})"

    def __format__(self, format_spec):
        """格式化 —— format(obj, spec) 调用"""
        if format_spec == "polar":
            import math
            r = math.hypot(self.x, self.y)
            theta = math.degrees(math.atan2(self.y, self.x))
            return f"(r={r:.2f}, θ={theta:.1f}°)"
        return str(self)

    # ── 2. 比较运算 ──
    def __eq__(self, other):
        if isinstance(other, Vector2D):
            return self.x == other.x and self.y == other.y
        return NotImplemented   # 返回 NotImplemented 让 Python 尝试反向操作

    def __lt__(self, other):
        """按模长比较"""
        if isinstance(other, Vector2D):
            return (self.x**2 + self.y**2) < (other.x**2 + other.y**2)
        return NotImplemented

    def __hash__(self):
        """需要 hash 时必须与 __eq__ 一致：相等对象哈希值相同"""
        return hash((self.x, self.y))

    # ── 3. 算术运算 ──
    def __add__(self, other):
        if isinstance(other, Vector2D):
            return Vector2D(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector2D):
            return Vector2D(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar):
        """数乘：v * 3"""
        if isinstance(scalar, (int, float)):
            return Vector2D(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        """反向数乘：3 * v"""
        return self.__mul__(scalar)

    def __neg__(self):
        """取反：-v"""
        return Vector2D(-self.x, -self.y)

    def __abs__(self):
        """模长：abs(v)"""
        import math
        return math.hypot(self.x, self.y)

    # ── 4. 容器协议 ──
    def __len__(self):
        """维度数"""
        return 2

    def __getitem__(self, index):
        """索引访问：v[0], v[1]"""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Vector2D 只有 2 个分量")

    def __iter__(self):
        """迭代：for x in v"""
        yield self.x
        yield self.y

    # ── 5. 可调用对象 ──
    def __call__(self, other):
        """点积：v1(v2)"""
        if isinstance(other, Vector2D):
            return self.x * other.x + self.y * other.y
        raise TypeError("参数必须是 Vector2D")

    # ── 6. 上下文管理器 ──
    def __enter__(self):
        print(f"进入上下文: {self}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"退出上下文: {self}")
        if exc_type:
            print(f"  捕获异常: {exc_type.__name__}")
        return False   # 不抑制异常

# ── 使用演示 ──
v1 = Vector2D(3, 4)
v2 = Vector2D(1, 2)

print(repr(v1))           # "Vector2D(3, 4)"
print(str(v1))            # "(3, 4)"
print(format(v1, "polar")) # "(r=5.00, θ=53.1°)"

print(v1 + v2)            # (4, 6)
print(v1 * 3)             # (9, 12)
print(3 * v1)             # (9, 12) — __rmul__
print(abs(v1))            # 5.0
print(v1(v2))             # 11 — 点积 (3*1 + 4*2)
print(v1 == Vector2D(3, 4))  # True
print(v1 > v2)            # True (25 > 5)

# 作为字典键（需要 __hash__）
vectors = {v1: "vector1", v2: "vector2"}
print(vectors[Vector2D(3, 4)])   # "vector1"

# 上下文管理器
with Vector2D(1, 1) as v:
    print(f"使用中: {v}")

if __name__ == "__main__":
    print("OK")
