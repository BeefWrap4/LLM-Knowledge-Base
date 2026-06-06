# ---
# chapter: 05
# topic: Python并发编程
# section: 5.3.3 线程池 ThreadPoolExecutor
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 06_thread_pool_executor.py
# expected_runtime: ~2s
# expected_output: shows map() and submit()/as_completed() patterns
# ---
# See: ../tutorial/05_Python并发编程.md#533-线程池-threadpoolexecutor
# Interview hooks:
#   - ThreadPoolExecutor.map() 和 submit() 的区别？
#   - as_completed() 的工作原理？为什么要用它？
#   - 线程池的 max_workers 应该设多大？和 IO 等待时间的关系？
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def fetch_url(url):
    """模拟网络请求"""
    time.sleep(0.2)  # 模拟网络延迟
    return f"Response from {url}"


urls = [f"https://api.example.com/data/{i}" for i in range(10)]

# ========== 方式1：map（按顺序返回结果）==========
print("=== map 方式 ===")
with ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(fetch_url, urls)
    for url, result in zip(urls, results):
        print(f"{url}: {result}")

# ========== 方式2：submit（按完成顺序返回，更灵活）==========
print("\n=== submit 方式 ===")
with ThreadPoolExecutor(max_workers=5) as executor:
    # 提交所有任务，得到 Future 对象
    future_to_url = {executor.submit(fetch_url, url): url for url in urls}

    # as_completed 在任务完成时 yield
    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            result = future.result()
            print(f"OK {url}: {result}")
        except Exception as e:
            print(f"FAIL {url}: {e}")

if __name__ == "__main__":
    print("OK")
