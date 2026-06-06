# ---
# chapter: 1
# topic: 虚拟环境管理 venv vs conda
# section: 1.1.4
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 06_venv_conda.py
# expected_runtime: <1s
# expected_output: 命令序列说明
# ---
# See: ../tutorial/01_Python编程基础.md (lines 188-236)
# Interview hooks:
#   1. venv 和 conda 的核心区别?
#   2. 什么时候应该用 poetry 而不是 venv?
#   3. 为什么数据科学项目推荐 conda?
"""
虚拟环境管理：venv vs conda 对比
"""

# ┌─────────────────────────────────────────────────────────────┐
# │                    虚拟环境工具选择                           │
# ├─────────────────────────────────────────────────────────────┤
# │                                                             │
# │   纯 Python 项目 ──────→ python -m venv .venv              │
# │   (Web后端/脚本)           标准库内置，轻量                   │
# │                                                             │
# │   数据科学/AI 项目 ────→ conda create -n myenv python=3.12 │
# │   (NumPy/PyTorch)        管理非 Python 依赖（CUDA等）        │
# │                                                             │
# │   生产部署 ────────────→ poetry / pipenv                   │
# │                            精确锁定依赖版本                   │
# │                                                             │
# └─────────────────────────────────────────────────────────────┘

# venv 标准用法
def setup_venv(project_dir: str) -> None:
    """创建并激活虚拟环境的标准流程"""
    commands = [
        f"cd {project_dir}",
        "python -m venv .venv",                    # 创建环境
        "source .venv/bin/activate",               # Linux/Mac 激活
        # ".venv\\Scripts\\activate",              # Windows 激活
        "pip install --upgrade pip",
        "pip install -r requirements.txt",
    ]
    print("执行命令序列：")
    for cmd in commands:
        print(f"  $ {cmd}")

# conda 环境管理（数据科学项目推荐）
def setup_conda(env_name: str, python_version: str = "3.12") -> None:
    """创建 conda 环境的标准流程"""
    commands = [
        f"conda create -n {env_name} python={python_version} -y",
        f"conda activate {env_name}",
        "conda install numpy pandas pytorch -c pytorch",  # 安装带 CUDA 的 PyTorch
    ]
    print("执行命令序列：")
    for cmd in commands:
        print(f"  $ {cmd}")


if __name__ == "__main__":
    setup_venv("./my_project")
    print()
    setup_conda("data_science", "3.12")
    print("OK")
