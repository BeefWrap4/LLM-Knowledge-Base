# ---
# chapter: 16
# topic: 模型微调与推理优化
# section: 16.6.3 Xinference 多模型管理部署
# difficulty: ⭐⭐⭐
# tier: gpu
# deps: xinference (or fallback to local mock client)
# run: python 07_xinference_deployment.py --mock
# expected_runtime: <5s for mock / 需先启动 xinference-local 服务
# expected_output: mock 模式打印 Xinference 接口演示 + 与 vLLM 直接部署的对比
# ---
# See: ../tutorial/16_模型微调与推理优化.md §16.6.3
# Interview hooks:
#   1. Xinference 相对 vLLM 的核心优势？多模型统一管理与 Web UI？
#   2. launch_model / get_model / terminate_model 的资源生命周期如何管理？
#   3. 什么场景适合选 Xinference vs 直接 vLLM 部署？

"""
Xinference 部署示例 —— 2026年推荐的多模型管理方案

启动: xinference-local --host 0.0.0.0 --port 9997
安装: pip install xinference
"""

import os
import argparse


MOCK_MODE = os.environ.get("MOCK_MODE", "0") == "1"


def mock_xinference():
    """Mock 模式演示"""
    print("[MOCK] Xinference 部署流程")
    print()
    print("  1) 启动服务: xinference-local --host 0.0.0.0 --port 9997")
    print("  2) 客户端连接 Xinference")
    print("  3) launch_model 部署指定模型 (自动下载 + 启动推理)")
    print("  4) get_model 拿到模型句柄, 调用 chat/generate")
    print("  5) terminate_model 释放资源")
    print()
    print("=" * 60)
    print("Xinference vs vLLM 直接部署")
    print("=" * 60)
    print("""
    维度         vLLM 直接部署          Xinference
    ---------  ------------------  --------------------
    模型管理     手动管理每个模型      统一管理, Web UI
    多模型       每个模型一个服务      一个平台多模型
    自动扩缩     需配合 K8s HPA       内置自动扩缩容
    适用规模     大规模生产           中小规模, 快速迭代
    上手难度     中 (需配置)           低 (一键启动)
    """)
    print()
    print("OK")


def real_xinference():
    """真实 Xinference 调用（需运行中的 xinference-local 服务）"""
    try:
        from xinference.client import Client
    except ImportError:
        print("未安装 xinference, 请先 pip install xinference")
        return

    client = Client("http://localhost:9997")

    # 列出可用的内置模型
    print("可用模型:", client.list_models())

    # 部署模型
    model_uid = client.launch_model(
        model_name="qwen2.5-instruct",
        model_size_in_billions=7,
        model_format="pytorch",
        quantization="none",
        n_gpu=1,
    )

    # 获取模型句柄, 推理
    model = client.get_model(model_uid)
    response = model.chat(
        messages=[{"role": "user", "content": "你好!"}],
        generate_config={"temperature": 0.7, "max_tokens": 512},
    )
    print("回复:", response["choices"][0]["message"]["content"])

    # 停止释放资源
    client.terminate_model(model_uid)
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or MOCK_MODE:
        mock_xinference()
    else:
        real_xinference()
