# ---
# chapter: 7
# topic: Python 并发编程
# topic_id: concurrency.threading_basics
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 04_threading_basics.py
# expected_runtime: ~2s
# expected_output: 3 worker threads complete in parallel, then main prints summary
# ---
# See: ../../../07_Python并发编程.md
# Interview hooks:
#   - threading.Thread 的 daemon 参数有什么作用？
#   - start() 和 run() 的区别是什么？
#   - join() 的作用？设置 timeout 会有什么行为？
import threading
import time


def worker(name, duration):
    """线程执行的任务"""
    print(f"[Thread-{name}] 开始执行")
    time.sleep(duration)  # 模拟 IO 操作
    print(f"[Thread-{name}] 执行完成，耗时 {duration}s")


# 创建并启动线程
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i, 2))
    threads.append(t)
    t.start()

# 等待所有线程完成
for t in threads:
    t.join()

print("所有线程执行完毕")

if __name__ == "__main__":
    print("OK")
