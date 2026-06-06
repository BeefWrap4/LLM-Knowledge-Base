# ---
# chapter: 05
# topic: Python并发编程
# section: 5.1 GIL 全局解释器锁
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 01_gil_switch_interval.py
# expected_runtime: <1s
# expected_output: prints GIL switch interval
# ---
# See: ../tutorial/05_Python并发编程.md#51-gil-全局解释器锁
# Interview hooks:
#   - GIL 是什么？为什么 CPython 需要 GIL？
#   - GIL 在什么时机释放？默认切换间隔是多少？
#   - 如何调整 GIL 切换间隔？调整后有什么影响？
import sys

# 查看 GIL 切换间隔（默认 5ms）
print(f"GIL switch interval: {sys.getswitchinterval()}s")
# 可以调整切换间隔
sys.setswitchinterval(0.01)  # 改为 10ms
print(f"Updated GIL switch interval: {sys.getswitchinterval()}s")

if __name__ == "__main__":
    print("OK")
