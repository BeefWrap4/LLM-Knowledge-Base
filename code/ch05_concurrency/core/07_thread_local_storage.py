# ---
# chapter: 05
# topic: Python并发编程
# section: 5.3.4 线程本地存储
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 07_thread_local_storage.py
# expected_runtime: ~0.5s
# expected_output: each thread prints its own user/request_id
# ---
# See: ../tutorial/05_Python并发编程.md#534-线程本地存储
# Interview hooks:
#   - threading.local() 的作用是什么？典型应用场景？
#   - 为什么 Web 框架（如 Flask）会用线程本地存储？
#   - threading.local 与 contextvars 的区别？协程场景下应该用哪个？
import threading
import time

# 线程本地存储：每个线程拥有独立的数据副本
thread_local = threading.local()

def process_request(request_id):
    # 每个线程的 thread_local.user 互不干扰
    thread_local.user = f"User-{request_id}"
    thread_local.request_id = request_id

    # 模拟处理
    time.sleep(0.05)

    print(f"线程 {threading.current_thread().name}: "
          f"user={thread_local.user}, request={thread_local.request_id}")

# 多线程场景下（如 Web 服务器），每个请求独立存储上下文
threads = []
for i in range(5):
    t = threading.Thread(target=process_request, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

if __name__ == "__main__":
    print("OK")
