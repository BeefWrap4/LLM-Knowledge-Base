# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.7.3 实时语音 - 全双工并发协议教学骨架
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# mock_safe: true
# deps: Python standard library
# run: python 10_moshi_realtime.py
# expected_runtime: <1s
# expected_output: 本地队列的并发发送/接收计数
# ---
# See: ../tutorial/21_多模态大模型.md#21-6-3-实时语音模型
# Interview hooks:
#   1. 全双工语音为何需要独立的上行、下行与取消控制？
#   2. 音频 codec 的帧率、码率与端到端延迟如何共同评测？
#   3. 真实客户端还需要哪些设备、鉴权、背压和重连机制？

import asyncio


async def protocol_structure_demo(chunk_count: int = 5) -> tuple[int, int]:
    """用本地队列演示并发方向；不连接 Moshi，也不处理真实音频。"""
    upstream: asyncio.Queue[bytes | None] = asyncio.Queue()
    downstream: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def send_user_audio() -> int:
        for index in range(chunk_count):
            await upstream.put(f"user-{index}".encode())
        await upstream.put(None)
        return chunk_count

    async def simulated_server() -> None:
        while (chunk := await upstream.get()) is not None:
            await downstream.put(b"model-" + chunk)
        await downstream.put(None)

    async def receive_model_audio() -> int:
        received = 0
        while await downstream.get() is not None:
            received += 1
        return received

    sent, _, received = await asyncio.gather(
        send_user_audio(),
        simulated_server(),
        receive_model_audio(),
    )
    return sent, received


def main() -> None:
    sent, received = asyncio.run(protocol_structure_demo())
    assert sent == received == 5
    print("[STRUCTURE ONLY] No Moshi server, codec, microphone, or speaker was used.")
    print(f"sent={sent}, received={received}")


if __name__ == "__main__":
    main()
    print("OK")
