# ---
# chapter: 05
# topic: Python并发编程
# section: 5.3.2 线程同步机制
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 05_threading_sync_mechanisms.py
# expected_runtime: ~3s
# expected_output: demonstrates Lock/RLock/Semaphore/Condition behaviors
# ---
# See: ../tutorial/05_Python并发编程.md#532-线程同步机制
# Interview hooks:
#   - Lock 和 RLock 的区别？什么场景必须用 RLock？
#   - Semaphore 与 Lock 的区别？Semaphore 的常见用途？
#   - Condition 变量的 wait/notify 机制如何避免"虚假唤醒"？
import threading
import time

# ========== Lock（互斥锁）==========
lock = threading.Lock()
counter = 0


def increment_with_lock(n):
    """使用 Lock 保证线程安全"""
    global counter
    for _ in range(n):
        with lock:  # 等价于 lock.acquire() + lock.release()
            # 临界区：读取-修改-写入操作
            current = counter
            time.sleep(0.000001)  # 模拟操作延迟
            counter = current + 1


# ========== RLock（可重入锁）==========
rlock = threading.RLock()


def outer():
    with rlock:
        print("外层获取锁")
        inner()  # 同一线程可以再次获取 RLock


def inner():
    with rlock:
        print("内层获取锁（重入）")


# RLock 允许同一线程多次获取，Lock 会死锁

# ========== Semaphore（信号量）==========
# 控制同时访问某资源的线程数量
semaphore = threading.Semaphore(3)  # 最多3个线程同时执行


def limited_worker(name):
    with semaphore:
        print(f"{name} 获取信号量，开始执行")
        time.sleep(0.2)
        print(f"{name} 释放信号量")


# ========== Condition（条件变量）==========
condition = threading.Condition()
message = None


def consumer():
    with condition:
        while message is None:
            condition.wait()  # 等待通知
        print(f"消费者收到: {message}")


def producer():
    global message
    time.sleep(0.5)
    with condition:
        message = "Hello"
        condition.notify_all()  # 通知所有等待的线程


# 运行演示
t1 = threading.Thread(target=increment_with_lock, args=(1000,))
t2 = threading.Thread(target=increment_with_lock, args=(1000,))
t1.start()
t2.start()
t1.join()
t2.join()
print(f"Lock 计数器最终: {counter} (期望 2000)")

outer()

# Semaphore 演示：5 个 worker 共享 3 个槽位
sem_threads = [threading.Thread(target=limited_worker, args=(f"W{i}",)) for i in range(5)]
for t in sem_threads:
    t.start()
for t in sem_threads:
    t.join()

# Condition 演示
p = threading.Thread(target=producer)
c = threading.Thread(target=consumer)
c.start()
p.start()
p.join()
c.join()

if __name__ == "__main__":
    print("OK")
