.PHONY: help install-core install-llm install-gpu install-all test test-llm test-gpu lint fmt clean run verify verify-xrefs sync-links run-all ci ci-quick ci-core ci-llm download-models llm-doctor docker-build docker-up docker-down docker-bash docker-llm docker-gpu setup-local integration-test

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

sync-links:  ## 检查教程↔代码双向链接 (§X.Y ↔ # section: X.Y)
	python scripts/sync_links.py

llm-doctor:  ## 诊断已配置 API Key 的厂商 (deepseek/kimi/siliconflow/openai)
	python scripts/llm_doctor.py

download-models:  ## 从国内源下载教程所需模型 (默认: embedding + reranker, ~700MB)
	python scripts/download_models.py $${ARGS}

run-all:  ## 跑完所有 core/ 例子 (TIER=core|llm|gpu, PARALLEL=8)
	python scripts/run_all_examples.py --tier $${TIER:-core} --parallel $${PARALLEL:-8}

# ─────────────────────────────────────────────────────────
# CI 集成 (Wave 10)
# ─────────────────────────────────────────────────────────

ci:  ## 完整 CI: quick + core + llm (~10 min, 镜像 GitHub Actions)
	@echo "=== [1/3] Quick check (wiki / README / smoke) ==="
	@python scripts/verify_all.py
	@echo ""
	@echo "=== [2/3] Core tier (158 examples) ==="
	@python scripts/run_all_examples.py --tier core --parallel 4 --timeout 60
	@echo ""
	@echo "=== [3/3] LLM tier (199 examples, mock mode) ==="
	@python scripts/run_all_examples.py --tier llm --parallel 4 --timeout 180

ci-quick:  ## 仅 quick 检查 (wiki / README / smoke) — <30s
	python scripts/verify_all.py

ci-core:  ## 仅 core tier — ~30s
	python scripts/run_all_examples.py --tier core --parallel 4 --timeout 60

ci-llm:  ## 仅 llm tier — ~3 min
	python scripts/run_all_examples.py --tier llm --parallel 4 --timeout 180

# ─────────────────────────────────────────────────────────
# Docker (Wave 14-C)
# ─────────────────────────────────────────────────────────

docker-build:  ## 构建 Docker 镜像 (默认 core + llm tier, 国内源)
	docker build -t llm-kb:latest \
		--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
		--build-arg REGISTRY=docker.io \
		-t llm-kb:latest .

docker-up:  ## 启动 core profile (仅 app 容器)
	cd .. && docker compose --profile core up -d

docker-llm:  ## 启动 llm profile (app + Redis)
	cd .. && docker compose --profile llm up -d

docker-gpu:  ## 启动 gpu profile (app + Redis + pgvector, 需 NVIDIA GPU)
	cd .. && docker compose --profile gpu up -d

docker-bash:  ## 进 core 容器交互式 bash
	cd .. && docker compose --profile core run --rm app bash

docker-down:  ## 停所有容器
	cd .. && docker compose --profile core --profile llm --profile gpu down

docker-ci:  ## 在容器内跑 make ci-quick (验证 Docker 镜像完整性)
	cd .. && docker compose --profile core run --rm app make -C code ci-quick

# ─────────────────────────────────────────────────────────
# 本地一键部署 (Wave 27)
# ─────────────────────────────────────────────────────────

setup-local:  ## 一键本地部署 (装依赖 + 启中间件 + 下模型 + 集成测试, ~5-10 min)
	bash scripts/setup_local.sh

setup-local-quick:  ## 快速版 (跳过模型下载, ~30s)
	bash scripts/setup_local.sh --skip-models

integration-test:  ## 跑 4 项集成测试 (Embedding + Redis + pgvector + LLM)
	python scripts/test_integration.py

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
