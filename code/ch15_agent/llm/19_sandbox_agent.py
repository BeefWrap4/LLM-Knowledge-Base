# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.9.4 SandboxAgent - 隔离工作区
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: [openai-agents>=0.14.0]  # --check-sdk 模式需要
# run: python 19_sandbox_agent.py
# expected_runtime: 离线 <1s（默认配置检查）
# expected_output: Manifest/SandboxAgent/SandboxRunConfig 责任边界
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.9.4-SandboxAgent
# Interview hooks:
#   1. Manifest、SandboxAgent、SandboxRunConfig 分别负责什么？
#   2. 为什么 Manifest 不是网络、资源或审批策略的替代品？
#   3. Windows 本地开发为什么优先选 Docker 或托管 sandbox client？
"""
OpenAI Agents SDK Sandbox Agents（beta）当前 API 形状。

默认模式不调用模型、不创建 sandbox，只校验配置责任边界。显式 ``--check-sdk``
会导入 ``agents.sandbox`` 并构造对象，但仍不会调用模型；真实运行必须由应用注入
一个受支持且已经配置好的 sandbox client。
"""

import argparse
from typing import Any


def offline_plan() -> dict[str, Any]:
    """返回与官方 API 分层一致、但不会冒充 SDK 对象的离线计划。"""
    return {
        "agent": {
            "type": "SandboxAgent",
            "default_manifest": {
                "type": "Manifest",
                "entries": ["task.md", "output/"],
            },
        },
        "run": {
            "type": "SandboxRunConfig",
            "client": "injected BaseSandboxClient",
        },
        "boundary": (
            "Manifest 描述新工作区内容；SandboxAgent 描述角色与默认工作区；"
            "SandboxRunConfig 在每次运行时选择 client/session/snapshot。"
        ),
    }


def build_sdk_objects(model: str, client=None):
    """按当前官方导入构建 SDK 对象；client=None 时只用于静态检查。"""
    try:
        from agents.run import RunConfig
        from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
        from agents.sandbox.entries import Dir, File
    except ImportError as exc:
        raise RuntimeError(
            "SDK 检查需要支持 Sandbox Agents 的 `openai-agents>=0.14.0`；"
            "当前环境可能仍是旧版。"
        ) from exc

    manifest = Manifest(
        entries={
            "task.md": File(content=b"Write a short answer to output/result.txt."),
            "output": Dir(),
        }
    )
    agent = SandboxAgent(
        name="Sandbox writer",
        model=model,
        instructions="Read task.md, write the result under output/, then summarize the verification.",
        default_manifest=manifest,
    )
    run_config = RunConfig(
        sandbox=SandboxRunConfig(client=client),
        workflow_name="Sandbox API tutorial",
    )
    return agent, run_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-sdk", action="store_true", help="导入当前 SDK 并构造对象")
    parser.add_argument("--model", default="gpt-5.6-sol")
    args = parser.parse_args()

    if args.check_sdk:
        agent, run_config = build_sdk_objects(args.model)
        print(type(agent).__name__)
        print(type(run_config.sandbox).__name__)
        print("[check] 未注入 client，不执行 Runner")
    else:
        plan = offline_plan()
        print(plan["agent"])
        print(plan["run"])
        print(plan["boundary"])
    print("OK")


if __name__ == "__main__":
    main()
