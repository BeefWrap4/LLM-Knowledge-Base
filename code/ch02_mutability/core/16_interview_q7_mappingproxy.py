# ---
# chapter: 2
# topic: 面试 Q7 — Python 只读字典 (MappingProxyType)
# section: Q7
# difficulty: ⭐⭐
# tier: core
# deps: types
# run: python 16_interview_q7_mappingproxy.py
# expected_runtime: <1s
# expected_output: 演示 MappingProxyType 视图 + 视图下标可读不可写
# ---
# See: ../tutorial/02_Python核心面试专题_可变性与拷贝.md#q7python-中如何实现只读字典
# Interview hooks:
#   1. Python 中如何实现只读字典?三种方式?
#   2. MappingProxyType 是"复制"还是"视图"?性能差异?
#   3. 通过视图修改原 dict 会影响视图吗?为什么?

"""
面试真题 Q7:Python 中如何实现只读字典?
答案:
    1. types.MappingProxyType(dict) — 推荐,不复制数据
    2. 自定义类封装 dict
    3. 冻结后用深拷贝分发
"""

from types import MappingProxyType

data = {"a": 1, "b": [2, 3]}
read_only = MappingProxyType(data)
print(read_only["a"])   # 1
# read_only["a"] = 2    # TypeError: 'mappingproxy' object does not support item assignment
# 注意:data["b"].append(4) 仍然会影响 read_only,因为只是视图

if __name__ == "__main__":
    print("OK")
