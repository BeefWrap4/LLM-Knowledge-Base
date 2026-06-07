# ---
# chapter: 05
# topic: Python并发编程
# section: 5.4.2 进程间通信（IPC）
# difficulty: ⭐⭐⭐
# tier: core
# deps: []
# run: python 09_multiprocessing_ipc.py
# expected_runtime: ~3s
# expected_output: Queue, Pipe, shared-memory demos each run in sequence
# ---
# See: ../tutorial/05_Python并发编程.md#542-进程间通信ipc
# Interview hooks:
#   - Queue / Pipe / shared memory 各自的适用场景？
#   - 为什么 Queue.put 必须配合 join 或 task_done？
#   - mp.Value 和 mp.Array 跨进程安全吗？为什么仍需要 Lock？
import multiprocessing as mp
import time


def producer(queue, items):
    """生产者：通过 Queue 发送数据"""
    for item in items:
        queue.put(item)
        print(f"[生产者] 发送: {item}")
        time.sleep(0.05)
    queue.put(None)  # 发送结束信号


def consumer(queue):
    """消费者：从 Queue 接收数据"""
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"[消费者] 处理: {item}")


def run_queue_demo():
    queue = mp.Queue(maxsize=10)
    items = list(range(10))
    p1 = mp.Process(target=producer, args=(queue, items))
    p2 = mp.Process(target=consumer, args=(queue,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


# ========== 使用 Pipe（双向通信）==========
def send_data(conn, data):
    conn.send(data)
    conn.close()


def recv_data(conn):
    print(f"收到: {conn.recv()}")
    conn.close()


def run_pipe_demo():
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=send_data, args=(child_conn, "Hello"))
    p.start()
    print(parent_conn.recv())  # 输出: Hello
    p.join()


# ========== 使用共享内存（Value / Array）==========
def increment(shared_counter, lock, n):
    """使用共享内存 + 锁实现进程安全计数"""
    for _ in range(n):
        with lock:
            shared_counter.value += 1


def run_shared_memory_demo():
    shared_counter = mp.Value('i', 0)  # 'i' = signed int
    lock = mp.Lock()

    processes = [
        mp.Process(target=increment, args=(shared_counter, lock, 2500))
        for _ in range(4)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f"最终计数: {shared_counter.value}")  # 10000


def main():
    print("=== Queue Demo ===")
    run_queue_demo()
    print("\n=== Pipe Demo ===")
    run_pipe_demo()
    print("\n=== Shared Memory Demo ===")
    run_shared_memory_demo()


if __name__ == "__main__":
    main()
