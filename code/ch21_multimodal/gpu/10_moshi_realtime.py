# ---
# chapter: 21
# topic: 多模态大模型
# section: 21.6.3 实时语音模型 - Moshi 全双工对话
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: asyncio (真实模式需 moshi + websockets)
# run: python 10_moshi_realtime.py
# expected_runtime: <5s (mock)
# expected_output: Moshi 全双工对话流式协议演示
# ---
# See: ../tutorial/21_多模态大模型.md#21-6-3-实时语音模型
# Interview hooks:
#   1. 全双工语音对话相比 ASR+LLM+TTS pipeline 延迟能降低多少？
#   2. Mimi codec 为什么能做到 12.5Hz 超低码率？
#   3. Inner Monologue 机制如何提升语音回复的语义连贯性？

import asyncio
import os


async def main_async():
    """演示 Moshi 全双工流式对话（mock）。"""
    use_mock = os.environ.get("CH21_MOCK", "1") == "1"

    if use_mock:
        # 模拟全双工流：发送用户音频 + 接收模型音频
        send_count = 0
        recv_count = 0
        max_chunks = 5  # 演示用，只跑 5 个 chunk

        async def send_audio():
            nonlocal send_count
            for i in range(max_chunks):
                chunk = f"user_audio_chunk_{i}"
                send_count += 1
                await asyncio.sleep(0.01)
            return send_count

        async def recv_audio():
            nonlocal recv_count
            for i in range(max_chunks):
                chunk = f"model_audio_chunk_{i}"
                recv_count += 1
                await asyncio.sleep(0.01)
            return recv_count

        # 真实场景下会使用 MoshiClient + 麦克风/扬声器
        results = await asyncio.gather(send_audio(), recv_audio())
        print(f"Sent chunks: {results[0]}, Received chunks: {results[1]}")
        print("Moshi full-duplex demo OK")
        return

    # 真实模式
    try:
        from moshi import MoshiClient
    except ImportError:
        print("moshi package not installed. Skipping real mode.")
        return

    client = MoshiClient("ws://localhost:8998")

    async def mic_stream():
        """占位麦克风流: 真实部署时替换为 PyAudio/sounddevice 捕获"""
        if False:
            yield b""
        return

    async def speaker_play(chunk: bytes) -> None:
        """占位扬声器播放: 真实部署时替换为 sounddevice 输出"""
        _ = chunk

    async def send_audio():
        async for chunk in mic_stream():
            await client.send_user_audio(chunk)

    async def recv_audio():
        async for chunk in client.recv_model_audio():
            await speaker_play(chunk)

    await asyncio.gather(send_audio(), recv_audio())


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
