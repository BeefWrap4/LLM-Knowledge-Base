# ---
# chapter: 28
# topic: WebLLM 浏览器端推理 (WebGPU 加速)
# section: 28.5.1 WebLLM / MLC-LLM
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: (浏览器侧: @mlc-ai/web-llm; 本文件仅展示 JS 调用模板)
# run: python 07_webllm_browser_inference.py
# expected_runtime: <1s
# expected_output: WebLLM 完整前端调用代码 + 配套 Python 后端代理示例
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.5.1
# Interview hooks:
#   1. WebLLM 的运行原理是什么?为什么能直接跑在浏览器?
#   2. WebGPU 相比 WebAssembly 在 LLM 推理上的性能差距?
#   3. 浏览器推理的工程挑战 (模型分发/缓存/版本)?
"""WebLLM 浏览器推理示例: 前端调用模板 + 配套 Python 后端代理."""
from __future__ import annotations

WEBLLM_JS_TEMPLATE = '''
// 1. 安装: npm i @mlc-ai/web-llm
// 2. 浏览器需要 Chromium 113+ 或 Safari 17+ 才能用 WebGPU

import * as webllm from "@mlc-ai/web-llm";

const appConfig = {
  model_list: [
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
      model_id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
      model_type: "llama",
      vram_required_MB: 2048,
      // 量化等级: q4f16_1 = 4-bit 权重量化, FP16 激活
    },
  ],
};

async function initEngine() {
  const engine = await webllm.CreateMLCEngine(
    "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    {
      appConfig,
      initProgressCallback: (report) => {
        console.log(\`[WebLLM] \${report.text} (\${report.progress.toFixed(1)}%)\`);
        // report.text 例子: "Loading model weights..."
      },
    }
  );
  return engine;
}

async function chat(userMessage) {
  const engine = await initEngine();
  const reply = await engine.chat.completions.create({
    messages: [{ role: "user", content: userMessage }],
    temperature: 0.7,
    max_tokens: 256,
  });
  return reply.choices[0].message.content;
}

// 流式响应
async function chatStream(userMessage, onChunk) {
  const engine = await initEngine();
  const stream = await engine.chat.completions.create({
    messages: [{ role: "user", content: userMessage }],
    stream: true,
  });
  for await (const chunk of stream) {
    onChunk(chunk.choices[0]?.delta?.content || "");
  }
}

chat("Hello!").then(console.log);
'''


def browser_challenge_summary() -> None:
    """浏览器推理的工程挑战."""
    print("--- 浏览器推理挑战 ---")
    challenges = [
        ("首次加载",      "1-5GB 模型需下载, 需 Service Worker 缓存"),
        ("GPU 显存",     "WebGPU 通常限制 4-8GB, 7B Q4 是上限"),
        ("浏览器兼容",   "Chrome/Edge 113+, Safari 17+, Firefox 实验"),
        ("Tab 关闭即停", "无后台运行, PWA 需 Workaround"),
        ("模型分发",     "推荐 MLC 格式 (TVM 编译) 而非 GGUF, 因为更小"),
        ("DRM/版权",     "权重需 CDN 签名, 防未授权分发"),
    ]
    for tag, desc in challenges:
        print(f"  {tag}: {desc}")


def supported_models() -> None:
    """WebLLM 支持的常见模型."""
    print("\n--- WebLLM 支持的模型 (MLC-AI 编译) ---")
    models = [
        ("Llama-3.2-1B-Instruct-q4f16_1-MLC",  "1B", "1.0GB", "入门, 移动端"),
        ("Llama-3.2-3B-Instruct-q4f16_1-MLC",  "3B", "1.8GB", "⭐ 浏览器黄金标准"),
        ("Phi-3.5-mini-instruct-q4f16_1-MLC",  "3.8B", "2.3GB", "微软小模型"),
        ("Qwen2.5-7B-Instruct-q4f16_1-MLC",    "7B", "4.1GB", "中文好, 显存紧"),
        ("gemma-2-2b-it-q4f16_1-MLC",          "2B", "1.4GB", "Google 开源"),
    ]
    print(f"{'模型 ID':<45} {'大小':<6} {'体积':<8} {'备注'}")
    print("-" * 80)
    for mid, sz, vol, note in models:
        print(f"{mid:<45} {sz:<6} {vol:<8} {note}")


def main() -> None:
    print("=== WebLLM 浏览器端推理模板 ===\n")
    print(WEBLLM_JS_TEMPLATE)
    browser_challenge_summary()
    supported_models()


if __name__ == "__main__":
    main()
