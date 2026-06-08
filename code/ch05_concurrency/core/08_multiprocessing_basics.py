# ---
# chapter: 05
# topic: Python并发编程
# section: 5.4.1 基础用法
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 08_multiprocessing_basics.py
# expected_runtime: ~5s
# expected_output: Pool + Process demos both print fib results
# ---
# See: ../tutorial/05_Python并发编程.md#541-基础用法
# Interview hooks:
#   - 为什么 Windows 上 multiprocessing 必须有 if __name__ == "__main__" 保护？
#   - mp.Pool 和手动创建 mp.Process 的取舍？
#   - 进程间通信的几种方式及其适用场景？
import multiprocessing as mp
import os


def cpu_intensive_task(n):
    """CPU 密集型任务：计算斐波那契数列"""
    print(f"进程 {os.getpid()} 处理 n={n}")

    def fib(n):
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

    result = fib(n)
    return result


def run_pool_demo():
    numbers = [30, 32, 33, 31, 30]

    # ========== 方式1：Pool（进程池）==========
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # map 会自动分配任务到多个进程
        results = pool.map(cpu_intensive_task, numbers)
        print(f"Pool 结果: {results}")


def run_process_demo():
    numbers = [30, 32, 33, 31, 30]
    # ========== 方式2：Process（手动创建）==========
    processes = []
    for n in numbers[:3]:
        p = mp.Process(target=cpu_intensive_task, args=(n,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def main():
    # Windows 必须使用 if __name__ == "__main__" 保护
    run_pool_demo()
    run_process_demo()


if __name__ == "__main__":
    main()
