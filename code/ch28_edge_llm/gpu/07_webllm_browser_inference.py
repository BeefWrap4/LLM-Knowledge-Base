# ---
# chapter: 28
# topic: WebLLM 浏览器端推理 (真实 playwright 打开 mlc.ai/web-llm)
# section: 28.5.1 WebLLM / MLC-LLM
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: playwright (chromium)
# run: python 07_webllm_browser_inference.py
# expected_runtime: 30-90s (页面加载 + 浏览器下载模型 1-2GB)
# expected_output: 截屏保存到 code/.benchmarks/webllm-demo.png + 提取响应文本
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.5.1
# Interview hooks:
#   1. WebLLM 的运行原理是什么?为什么能直接跑在浏览器?
#   2. WebGPU 相比 WebAssembly 在 LLM 推理上的性能差距?
#   3. 浏览器推理的工程挑战 (模型分发/缓存/版本)?
"""WebLLM 浏览器推理示例: 真实用 playwright 打开 mlc.ai/web-llm, 输入 prompt, 截屏."""

from __future__ import annotations

import sys
from pathlib import Path

# 让脚本既能 `python file.py` 也能 `import` 找到 shared/
_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help
from shared.gpu_guard import skip_if_mock, skip_unless_enabled

# 浏览器侧 (供前端工程师复制) 的 JS 调用模板.
# 真实运行时, 模型权重由浏览器在客户端下载 + 编译, 无需服务端.
WEBLLM_JS_TEMPLATE = """\
// 浏览器侧调用: 完整前端代码, 复制到 .html 即可
// 1. npm i @mlc-ai/web-llm
// 2. 需要 Chromium 113+ 或 Safari 17+ 才能用 WebGPU
import * as webllm from "@mlc-ai/web-llm";

const appConfig = {
  model_list: [{
    model: "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
    model_id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    model_type: "llama",
    vram_required_MB: 2048,
  }],
};

const engine = await webllm.CreateMLCEngine(
  "Llama-3.2-3B-Instruct-q4f16_1-MLC",
  { appConfig, initProgressCallback: (r) => console.log(r.text, r.progress) }
);
const reply = await engine.chat.completions.create({
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(reply.choices[0].message.content);
"""


def run_webllm_in_browser(
    prompt: str = "Hello!",
    screenshot_path: str = "code/.benchmarks/webllm-demo.png",
    wait_ms: int = 15000,
) -> dict:
    """真实用 playwright 打开 webllm demo, 输入 prompt, 截屏, 提取响应.

    Args:
        prompt: 要发给 WebLLM 的用户消息.
        screenshot_path: 截屏保存路径 (绝对或相对 code/).
        wait_ms: 等待模型生成响应的毫秒数 (浏览器侧首次加载模型可能 30s+).

    Returns:
        dict 含 'screenshot' 和 'response' (提取到的 chat log 文本, 失败时为空).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise_with_help(
            "playwright 未装.",
            "运行 `pip install playwright && playwright install chromium`.",
        )

    out = Path(screenshot_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            # mlc.ai/web-llm 是官方 ChatBot demo 页面
            page.goto("https://mlc.ai/web-llm/", timeout=60000, wait_until="domcontentloaded")

            # 等待页面顶部标题加载 (避免立即操作)
            try:
                page.wait_for_selector("text=/WebLLM/i", timeout=15000)
            except Exception:
                # 不强制要求, 继续尝试找输入框
                pass

            # 尝试找到 chat 输入框 (页面有多个 fallback selector)
            input_selectors = [
                "textarea",
                "input[type='text']",
                "#user-input",
                "#chat-input",
            ]
            clicked = False
            for sel in input_selectors:
                try:
                    el = page.locator(sel).first
                    if el.count() > 0 and el.is_visible():
                        el.fill(prompt)
                        clicked = True
                        break
                except Exception:
                    continue

            # 提交: 优先按 Enter, 否则找发送按钮
            if clicked:
                try:
                    page.keyboard.press("Enter")
                except Exception:
                    try:
                        page.locator("button:has-text('Send')").first.click()
                    except Exception:
                        pass

            # 给浏览器一些时间下载/编译/推理
            page.wait_for_timeout(wait_ms)

            # 截屏保存
            page.screenshot(path=str(out), full_page=True)

            # 尝试提取响应文本
            response_text = ""
            for sel in ["#chat-log", "#chat-messages", ".chat-log", "main", "body"]:
                try:
                    text = page.text_content(sel) or ""
                    if text and len(text) > len(response_text):
                        response_text = text
                except Exception:
                    continue

            return {
                "screenshot": str(out),
                "response": response_text[:1000] if response_text else "(see screenshot)",
                "input_filled": clicked,
            }
        finally:
            browser.close()


def main() -> None:
    if skip_if_mock("已安装 Chromium 的 Playwright、WebGPU 和模型下载网络"):
        return
    if skip_unless_enabled(
        "WEBLLM_BROWSER_RUN",
        "the Chromium runtime, WebGPU support, model download size, and browser output path",
    ):
        return
    print("=== WebLLM 浏览器内推理 (真实 playwright) ===\n")
    print("JS 模板 (供前端复制):")
    print(WEBLLM_JS_TEMPLATE)
    print()

    result = run_webllm_in_browser("Hello!")
    print(f"截屏已保存: {result['screenshot']}")
    print(f"输入是否填入: {result['input_filled']}")
    print(f"提取的响应 (前 200 字):\n{result['response'][:200]}")
    print("OK")


if __name__ == "__main__":
    main()
