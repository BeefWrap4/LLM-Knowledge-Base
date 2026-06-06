.PHONY: help install-core install-llm install-gpu install-all test test-llm test-gpu lint fmt clean run verify verify-xrefs run-all

# ─────────────────────────────────────────────────────────
# 安装
# ─────────────────────────────────────────────────────────

help:  ## 显示所有命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-core:  ## 安装 core tier (30秒, CPU 即可)
	pip install -r requirements-core.txt

install-llm:  ## 安装 core + llm tier (5分钟)
	pip install -r requirements-llm.txt

install-gpu:  ## 安装 core + llm + gpu tier (30分钟, 需 NVIDIA GPU)
	pip install -r requirements-gpu.txt

install-all:  ## 安装所有 tier
	pip install -r requirements-gpu.txt

# ─────────────────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────────────────

test:  ## 跑 core tier smoke tests (默认, 笔记本 CI)
	pytest tests/ -m "not gpu" -q --tb=short

test-llm:  ## 跑 core + llm tier tests (用 mock_llm)
	pytest tests/ -m "not gpu" -q --tb=short

test-gpu:  ## 跑全部 tests, 包括 gpu tier (需 NVIDIA GPU)
	pytest tests/ -q --tb=short

# ─────────────────────────────────────────────────────────
# 同步验证 (Wave 5)
# ─────────────────────────────────────────────────────────

verify:  ## 教程↔代码 4 项检查 (wiki / README / code / smoke)
	python scripts/verify_all.py

verify-xrefs:  ## 只检查 wiki 链接完整性
	python scripts/verify_xrefs.py

run-all:  ## 跑完所有 core/ 例子 (TIER=core|llm|gpu, PARALLEL=8)
	python scripts/run_all_examples.py --tier $${TIER:-core} --parallel $${PARALLEL:-8}

# ─────────────────────────────────────────────────────────
# 代码质量
# ─────────────────────────────────────────────────────────

lint:  ## ruff 检查
	ruff check .

fmt:  ## ruff 自动修复 + 格式化
	ruff check --fix .
	ruff format .

typecheck:  ## mypy 类型检查
	mypy shared/ ch0*/ 2>&1 | head -50

# ─────────────────────────────────────────────────────────
# 运行
# ─────────────────────────────────────────────────────────

# 例子: make run FILE=ch12_transformer_architecture/core/01_scaled_dot_product_attention.py
run:  ## 跑指定 .py 文件 (FILE=path)
	python $(FILE)

# 例子: make chapter CH=12
chapter:  ## 列出某章所有 core/ 例子 (CH=12)
	@echo "─── Ch$(CH) core/ ───"
	@ls ch$(CH)_*/core/ 2>/dev/null || echo "无此章或无 core/ 例子"

clean:  ## 清理缓存
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/ __pycache__/ */__pycache__/ */*/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
