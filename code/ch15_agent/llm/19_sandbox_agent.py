# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.9.4 SandboxAgent - 隔离的代码执行环境
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 19_sandbox_agent.py
# expected_runtime: <1s（无 Docker 时演示静态分析）
# expected_output: 静态分析结果；docker 可用时执行沙箱代码
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.9.4-SandboxAgent
# Interview hooks:
#   1. 沙箱执行为什么用 Docker 而不是 Python subprocess + namespace？
#   2. 静态分析 + 资源限制 + 网络隔离是"纵深防御"，单层够不够？为什么？
#   3. 防御模式 (DANGEROUS_PATTERNS) 的局限是什么？有什么绕过方式？
"""
SandboxAgent - 隔离的代码执行环境
基于 OpenAI Agents SDK v0.14.0 沙箱模式
"""
import re
import shutil
import subprocess
import tempfile
import uuid


# 1. 基础沙箱 Agent 配置（实际由 OpenAI Agents SDK 提供）
SANDBOX_CONFIG = {
    "mode": "docker",
    "image": "python:3.12-slim",
    "memory_limit": "512m",
    "cpu_limit": "1.0",
    "network": "isolated",
    "allowed_domains": ["pypi.org"],
    "timeout_seconds": 30,
    "read_only_root": True,
}


# 2. 工具实现：受限容器内执行 Python
def python_executor(code: str) -> str:
    """
    在隔离 Docker 容器中执行 Python 代码
    """
    if not shutil.which("docker"):
        return "[Mock] docker 不可用，跳过实际执行；返回静态分析结果。"

    container_name = f"sandbox-{uuid.uuid4().hex[:8]}"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            code_path = f.name

        result = subprocess.run(
            [
                "docker", "run",
                "--name", container_name,
                "--rm",
                "-m", "512m",
                "--cpus", "1.0",
                "--network", "none",
                "-v", f"{code_path}:/tmp/code.py:ro",
                "--read-only",
                "--tmpfs", "/tmp:size=100m",
                "python:3.12-slim",
                "python", "/tmp/code.py",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return f"执行成功:\n{result.stdout}"
        else:
            return f"执行失败 (code={result.returncode}):\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "执行超时（30秒）"
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, timeout=5,
        )


# 3. 多层防御深度
class DefenseInDepth:
    """
    SandboxAgent 多层防御：
    1. 静态分析：扫描危险 API
    2. 资源限制：CPU 与内存与磁盘
    3. 网络隔离：默认全断
    4. 行为监控：异常行为检测
    """

    DANGEROUS_PATTERNS = [
        r"\bos\.system\b",
        r"\bsubprocess\b",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\b__import__\b",
        r"\bopen\s*\(.*['\"]/etc",
        r"\bopen\s*\(.*['\"]/proc",
    ]

    @classmethod
    def static_analysis(cls, code: str) -> tuple[bool, str]:
        """静态分析代码安全性"""
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                return False, f"检测到危险 API: {pattern}"
        return True, "静态分析通过"

    @classmethod
    def runtime_monitor(cls) -> dict:
        """运行时资源监控配置"""
        return {
            "memory": "512m",
            "cpu": "1.0",
            "pids_limit": 100,
            "network": "none",
            "read_only_fs": True,
            "no_new_privileges": True,
            "cap_drop": ["ALL"],
        }


def demo_sandbox():
    cases = [
        ("safe code", "import numpy as np\narr = np.array([1, 2, 3, 4, 5])\nprint(f'Mean: {arr.mean()}, Std: {arr.std()}')"),
        ("os.system", "import os\nos.system('rm -rf /tmp/important')"),
        ("eval", "eval('__import__(\"os\").system(\"whoami\")')"),
        ("read /etc", "open('/etc/passwd', 'r').read()"),
    ]

    for label, code in cases:
        print(f"=== {label} ===")
        safe, reason = DefenseInDepth.static_analysis(code)
        print(f"Static analysis: {safe} ({reason})")
        if safe:
            result = python_executor(code)
            print(f"Execution result: {result[:120]}")
        print()
    print("OK")


if __name__ == "__main__":
    demo_sandbox()
