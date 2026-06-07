# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.9.3 Strands BidiAgent
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: [strands-agents]  # 真实运行需要
# run: python 17_bidi_agent.py
# expected_runtime: 离线 <1s（mock）；真实依赖 strands SDK
# expected_output: 全双工语音 Agent 的关键配置
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.9.3-BidiAgent-与-Voice-Agent
# Interview hooks:
#   1. 什么叫"全双工"语音？和半双工（IVR）的本质区别？
#   2. VAD（Voice Activity Detection）的核心指标是什么？误唤醒怎么控制？
#   3. 语音 Agent 中"被打断"事件的处理时延要求？(200ms 以内)
"""
Strands BidiAgent - 双向语音对话示例
展示全双工音频流处理
"""
import asyncio


class AudioConfig:
    """音频配置（实际由 strands.voice.AudioConfig 提供）"""

    def __init__(self, input_sample_rate=16000, output_sample_rate=24000,
                 vad_sensitivity=0.6):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.vad_sensitivity = vad_sensitivity

    def __repr__(self):
        return (f"AudioConfig(in={self.input_sample_rate}Hz, "
                f"out={self.output_sample_rate}Hz, vad={self.vad_sensitivity})")


class BidiAgent:
    """简化版 BidiAgent（真实实现由 strands.BidiAgent 提供）"""

    def __init__(self, model, voice, system_prompt, audio_config):
        self.model = model
        self.voice = voice
        self.system_prompt = system_prompt
        self.audio_config = audio_config
        self.tools = {}

    def tool(self, fn):
        """装饰器：注册异步工具"""
        self.tools[fn.__name__] = fn
        return fn

    async def start_session(self, on_user_speech, on_agent_speech, on_interrupt):
        """启动会话（mock：等待用户输入模拟语音）"""
        print(f"[BidiAgent session started] model={self.model} voice={self.voice}")
        print(f"[audio] {self.audio_config}")
        print(f"[tools] {list(self.tools.keys())}")

        # 辅助函数: 同时支持 sync 和 async 回调
        async def _maybe_await(cb, *args):
            result = cb(*args)
            if hasattr(result, "__await__"):
                await result

        # 模拟一次完整对话轮次
        mock_user_text = "帮我查一下明天北京到上海的航班"
        await _maybe_await(on_user_speech, mock_user_text)

        # 工具调用
        if "search_flight" in self.tools:
            flight = await self.tools["search_flight"](
                origin="北京", destination="上海", date="2026-06-07"
            )
            await _maybe_await(on_agent_speech, f"为您找到航班 {flight['flight']}，票价 {flight['price']} 元。")
        else:
            await _maybe_await(on_agent_speech, "请告诉我出发地和目的地。")

        await _maybe_await(on_interrupt)  # 模拟用户打断


async def voice_assistant():
    """
    语音助手 BidiAgent
    特性：
    - 全双工：可被打断、随时插入
    - 流式：边说边处理
    - 多模态：语音加屏幕共享加工具调用
    """
    agent = BidiAgent(
        model="claude-4.6-realtime",
        voice="alloy",
        system_prompt="""你是一个友好的语音助手，名字叫小音。
        特点：
        - 回答简洁，语音场景不宜超过 30 字每句
        - 主动确认关键信息
        - 被打断时立即停止当前回答""",
        audio_config=AudioConfig(
            input_sample_rate=16000,
            output_sample_rate=24000,
            vad_sensitivity=0.6,
        ),
    )

    @agent.tool
    async def search_flight(origin: str, destination: str, date: str) -> dict:
        """查询航班信息"""
        return {"flight": "CA1234", "price": 580, "duration": "2h30m"}

    await agent.start_session(
        on_user_speech=lambda text: print(f"用户: {text}"),
        on_agent_speech=lambda text: print(f"小音: {text}"),
        on_interrupt=lambda: print("[用户打断]"),
    )


if __name__ == "__main__":
    asyncio.run(voice_assistant())
