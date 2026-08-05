# ---
# chapter: 7
# topic: Python 并发编程
# topic_id: concurrency.process_pool_executor
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 10_process_pool_executor.py
# expected_runtime: ~5s
# expected_output: prints prime count and first 10 primes
# ---
# See: ../../../07_Python并发编程.md
# Interview hooks:
#   - ProcessPoolExecutor 和 ThreadPoolExecutor 的取舍？
#   - 为什么 CPU 密集型任务首选 ProcessPoolExecutor？
#   - executor.map 和 submit 返回的 Future 在哪个进程中执行？
import math
from concurrent.futures import ProcessPoolExecutor


def is_prime(n):
    """判断素数（CPU 密集型）"""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def main():
    numbers = list(range(100000, 101000))

    # 使用进程池并行计算
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(is_prime, numbers))

    primes = [n for n, is_p in zip(numbers, results) if is_p]
    print(f"找到 {len(primes)} 个素数")
    print(f"前10个: {primes[:10]}")


if __name__ == "__main__":
    main()
    print("OK")
