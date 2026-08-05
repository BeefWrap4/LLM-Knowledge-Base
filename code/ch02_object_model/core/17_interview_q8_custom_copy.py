# ---
# chapter: 2
# topic: Python 对象模型与可变性
# topic_id: object_model.interview_q8_custom_copy
# difficulty: ⭐⭐⭐
# tier: core
# deps: copy
# run: python 17_interview_q8_custom_copy.py
# expected_runtime: <1s
# expected_output: 演示类同时实现 __copy__ 和 __deepcopy__ 两个钩子
# ---
# See: ../../../02_Python对象模型与可变性.md
# Interview hooks:
#   1. 如何让 copy.copy() 和 copy.deepcopy() 对同一类返回不同行为?
#   2. __copy__ 和 __deepcopy__(memo) 的签名区别?
#   3. __deepcopy__ 中 memo[id(self)] = new_obj 的作用是什么?

"""
面试真题 Q8:自定义类的拷贝行为
    — 如何让 copy.copy() 和 copy.deepcopy() 返回不同结果?
答案:分别实现 __copy__() 和 __deepcopy__(memo)
"""

import copy


class CustomCopy:
    def __init__(self, data):
        self.data = data
        self.id = id(self)

    def __copy__(self):
        """浅拷贝:共享 data"""
        new_obj = CustomCopy(self.data)  # 共享 data 引用
        return new_obj

    def __deepcopy__(self, memo):
        """深拷贝:复制 data"""
        new_data = copy.deepcopy(self.data, memo)
        new_obj = CustomCopy(new_data)
        memo[id(self)] = new_obj
        return new_obj


if __name__ == "__main__":
    # 演示
    original = CustomCopy([1, 2, 3])
    sh = copy.copy(original)
    dp = copy.deepcopy(original)
    print(f"浅拷贝共享 data? {sh.data is original.data}")  # True
    print(f"深拷贝独立 data? {dp.data is original.data}")  # False
    print("OK")
