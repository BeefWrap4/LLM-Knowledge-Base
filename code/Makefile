.PHONY: help install-core install-llm install-gpu install-all test test-llm test-llm-mock test-gpu test-real-api lint fmt clean run verify verify-xrefs sync-links run-all ci ci-quick ci-core ci-llm ci-real download-models download-models-list download-models-default download-models-llm download-models-gpu download-models-edge llm-doctor llm-doctor-setup llm-doctor-check docker-build docker-up docker-down docker-bash docker-llm docker-gpu setup-local integration-test

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

test-llm:  ## 跑 core + llm tier tests (mock 模式, 无需 API Key)
	LLM_MOCK=1 pytest tests/ -m "not gpu" -q --tb=short

test-llm-mock:  ## 跑 core + llm tier tests (强制 LLM_MOCK=1, 显式 mock 模式)
	LLM_MOCK=1 pytest tests/ -m "not gpu" -q --tb=short

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

ci-llm:  ## 仅 llm tier — ~3 min (mock 模式, 无需 API Key)
	LLM_MOCK=1 python scripts/run_all_examples.py --tier llm --parallel 4 --timeout 180

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

# ─────────────────────────────────────────────────────────
# LLM Doctor (Wave 28)
# ─────────────────────────────────────────────────────────

llm-doctor-setup:  ## 交互式配置 API Key
	python scripts/llm_doctor.py --setup

llm-doctor-check:  ## 测试已配置 Key
	python scripts/llm_doctor.py --check

ci-real:  ## 夜间跑真实 API 冒烟 (10 个代表性 LLM tier 文件, 需 DEEPSEEK_API_KEY)
	python scripts/test_real_api_smoke.py

test-real-api:  ## 真实 API 冒烟 (10 个代表性 LLM tier, 需 DEEPSEEK_API_KEY, 不可 LLM_MOCK=1)
	@test -n "$$DEEPSEEK_API_KEY" || (echo "[ERROR] DEEPSEEK_API_KEY 未设"; exit 1)
	@test "$$LLM_MOCK" != "1" || (echo "[ERROR] LLM_MOCK=1 与真实 API 互斥, 请 unset"; exit 1)
	unset LLM_MOCK && python scripts/test_real_api_smoke.py

# ─────────────────────────────────────────────────────────
# 模型下载 (Wave 28)
# ─────────────────────────────────────────────────────────

download-models-list:  ## 列出可下载的模型
	python scripts/download_models.py --list

download-models-default:  ## 下载默认必装模型 (embedding + reranker + qwen0.5b, ~1.7GB)
	python scripts/download_models.py --default

download-models-llm:  ## 下载 7B/8B LLM (vLLM 用, ~31GB)
	python scripts/download_models.py --tier llm-medium

download-models-gpu:  ## 下载 GPU 例子用模型 (cosmos / pi0 / r1-distill, ~25GB)
	python scripts/download_models.py --tier world-model --tier vla --tier reasoner

download-models-edge:  ## 下载端侧模型 (MLX / GGUF, ~7GB)
	python scripts/download_models.py --tier edge
