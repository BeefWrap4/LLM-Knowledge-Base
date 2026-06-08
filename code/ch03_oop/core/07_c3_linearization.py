# ---
# chapter: 3
# topic: C3 线性化算法原理
# section: 3.3.2 C3 线性化算法原理
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: 无
# run: python 07_c3_linearization.py
# expected_runtime: < 1s
# expected_output: OK
# ---
# See: ../tutorial/03_Python面向对象编程.md#3.3.2-C3-线性化算法原理
#
# Interview hooks:
# 1. C3 线性化的三个原则是什么？
# 2. 什么情况下 Python 会抛 MRO 冲突错误？举例说明。
# 3. 描述 C3 算法中 merge 的工作过程。

"""
C3 线性化算法 —— 方法解析顺序（MRO）的计算

C3 算法的三条原则：
1. 子类优先于父类
2. 多个父类按声明顺序
3. 单调性：如果在某个类的 MRO 中 A 在 B 前面，
   则该类的所有子类的 MRO 中 A 也在 B 前面

算法公式：
L(C) = C + merge(L(B1), L(B2), ..., [B1, B2, ...])
"""

# ─────────────────────────────────────────────────────────────
# MRO 计算示例
# ─────────────────────────────────────────────────────────────


class Base:
    pass


class X(Base):
    pass


class Y(Base):
    pass


class Z(X, Y):
    pass


print(f"Z 的 MRO: {[c.__name__ for c in Z.__mro__]}")
# ['Z', 'X', 'Y', 'Base', 'object']

"""
MRO 计算过程：

L(Z) = Z + merge(L(X), L(Y), [X, Y])
     = Z + merge([X, Base, object], [Y, Base, object], [X, Y])

merge 过程：
1. 取第一个列表的头 X，检查 X 不在其他列表的尾部
   → X 不在 [Base, object], [Base, object], [Y] 中
   → 可以取出！

   结果: [X] + merge([Base, object], [Y, Base, object], [Y])

2. 取第一个列表的头 Base，检查 Base 不在其他列表尾部
   → Base 在 [Base, object] 的尾部！不能取

3. 取第二个列表的头 Y，检查 Y 不在其他列表尾部
   → Y 不在 [object], [Base, object] 中
   → 可以取出！

   结果: [X, Y] + merge([Base, object], [Base, object])

4. 取第一个列表的头 Base
   → Base 不在 [object] 中
   → 可以取出！

最终结果: [Z, X, Y, Base, object]
"""

# ─────────────────────────────────────────────────────────────
# MRO 冲突（无法创建的情况）
# ─────────────────────────────────────────────────────────────

"""
以下继承关系会导致 MRO 冲突，Python 会抛出 TypeError：

class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass  # ✅ MRO: D -> B -> C -> A -> object

class E(C, B): pass  # ✅ MRO: E -> C -> B -> A -> object

# 但如果尝试同时继承 D 和 E：
# class F(D, E): pass  # ❌ TypeError: MRO 冲突！

原因：D 的 MRO 中 B 在 C 前面，但 E 的 MRO 中 C 在 B 前面，
      F 无法同时满足这两个顺序。
"""


# 验证
class A2:
    pass


class B2(A2):
    pass


class C2(A2):
    pass


class D2(B2, C2):
    pass


class E2(C2, B2):
    pass


try:

    class F2(D2, E2):
        pass
except TypeError as e:
    print(f"MRO 冲突: {e}")

if __name__ == "__main__":
    print("OK")
