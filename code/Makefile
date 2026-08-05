.PHONY: help install-core install-llm install-gpu install-all test test-llm test-llm-mock test-gpu test-real-api lint fmt clean run verify verify-xrefs sync-links run-all ci ci-quick ci-core ci-llm ci-real download-models download-models-list download-models-default download-models-llm download-models-gpu download-models-edge llm-doctor llm-doctor-setup llm-doctor-check docker-build docker-up docker-down docker-bash docker-llm docker-gpu setup-local integration-test vllm-server-start vllm-server-stop vllm-server-status

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

verify:  ## 10 项离线检查 (结构 / 来源 / 引用 / 统计 / smoke)
	python scripts/verify_all.py

verify-xrefs:  ## 只检查 wiki 链接完整性
	python scripts/verify_xrefs.py

sync-links:  ## 检查 section 引用完整性与代码覆盖率基线
	python scripts/sync_links.py

llm-doctor:  ## 仅显示 provider 配置与真实探针用法（不联网）
	python scripts/llm_doctor.py

download-models:  ## 按 ARGS 显式选择模型；下载体积以当前仓库元数据为准
	python scripts/download_models.py $${ARGS}

run-all:  ## 跑指定 tier (TIER=core|llm|gpu；gpu 默认 mock/skip, PARALLEL=8)
	python scripts/run_all_examples.py --tier $${TIER:-core} --parallel $${PARALLEL:-8}

# ─────────────────────────────────────────────────────────
# CI 集成 (Wave 10)
# ─────────────────────────────────────────────────────────

ci:  ## 完整 CI: quick + core + llm (~10 min, 镜像 GitHub Actions)
	@echo "=== [1/3] Quick check (10 offline gates) ==="
	@LLM_MOCK=1 python scripts/verify_all.py
	@echo ""
	@echo "=== [2/3] Core tier (158 examples) ==="
	@python scripts/run_all_examples.py --tier core --parallel 4 --timeout 60
	@echo ""
	@echo "=== [3/3] LLM tier (199 examples, mock mode) ==="
	@python scripts/run_all_examples.py --tier llm --parallel 4 --timeout 180

ci-quick:  ## 仅 10 项离线门禁 — <30s
	LLM_MOCK=1 python scripts/verify_all.py

ci-core:  ## 仅 core tier — ~30s
	python scripts/run_all_examples.py --tier core --parallel 4 --timeout 60

ci-llm:  ## 仅 llm tier — ~3 min (mock 模式, 无需 API Key)
	LLM_MOCK=1 python scripts/run_all_examples.py --tier llm --parallel 4 --timeout 180

# ─────────────────────────────────────────────────────────
# Docker (Wave 14-C)
# ─────────────────────────────────────────────────────────

docker-build:  ## 构建 Docker 镜像 (默认官方 PyPI；可用 PIP_INDEX_URL 覆盖)
	docker build -t llm-kb:latest \
		--build-arg PIP_INDEX_URL=$${PIP_INDEX_URL:-https://pypi.org/simple} \
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

setup-local:  ## 本地环境设置（会装依赖、启容器并可选下载模型）
	bash scripts/setup_local.sh

setup-local-quick:  ## 跳过模型下载；耗时仍取决于网络、缓存和 Docker
	bash scripts/setup_local.sh --skip-models

integration-test:  ## 真实集成；需 PROVIDER=<name>，会访问服务并产生 API 费用
	@test -n "$(PROVIDER)" || (echo "[ERROR] 必须显式设置 PROVIDER=<name>"; exit 2)
	RUN_REAL_INTEGRATION=1 LLM_MOCK=0 LLM_PROVIDER="$(PROVIDER)" python scripts/test_integration.py

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

# 例子: make run FILE=ch15_transformer/core/01_scaled_dot_product_attention.py
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

llm-doctor-check:  ## 真实 Key 探针；需 PROVIDER=<name>，可能产生费用
	@test -n "$(PROVIDER)" || (echo "[ERROR] 必须显式设置 PROVIDER=<name>"; exit 2)
	LLM_MOCK=0 LLM_PROVIDER="$(PROVIDER)" python scripts/llm_doctor.py --provider "$(PROVIDER)" --confirm-real

ci-real:  ## 严格真实 API 冒烟；需 PROVIDER=<name> 与对应 Key
	@test -n "$(PROVIDER)" || (echo "[ERROR] 必须显式设置 PROVIDER=<name>"; exit 2)
	RUN_REAL_INTEGRATION=1 LLM_MOCK=0 LLM_PROVIDER="$(PROVIDER)" python scripts/test_real_api_smoke.py --provider "$(PROVIDER)" --confirm-real

test-real-api:  ## 严格真实 API 冒烟；需 PROVIDER=<name> 与对应 Key
	@test -n "$(PROVIDER)" || (echo "[ERROR] 必须显式设置 PROVIDER=<name>"; exit 2)
	RUN_REAL_INTEGRATION=1 LLM_MOCK=0 LLM_PROVIDER="$(PROVIDER)" python scripts/test_real_api_smoke.py --provider "$(PROVIDER)" --confirm-real

# ─────────────────────────────────────────────────────────
# 模型下载 (Wave 28)
# ─────────────────────────────────────────────────────────

download-models-list:  ## 列出可下载的模型
	python scripts/download_models.py --list

download-models-default:  ## 下载当前清单中的必需模型
	python scripts/download_models.py --required-only

download-models-llm:  ## 下载中型 LLM；需显式确认大文件
	python scripts/download_models.py --llm-medium --confirm-large

download-models-gpu:  ## 下载 world model 与 reasoner；world model 需显式确认大文件
	python scripts/download_models.py --world-model --confirm-large
	python scripts/download_models.py --reasoner

download-models-edge:  ## 分别下载 MLX 与 GGUF 端侧模型
	python scripts/download_models.py --edge-mlx
	python scripts/download_models.py --edge-gguf

# ─────────────────────────────────────────────────────────
# Docker vLLM server (escape hatch for Windows, Wave 32+)
# ─────────────────────────────────────────────────────────

vllm-server-start:  ## 启动 vLLM Docker server (供本机 vLLM 例子连 OpenAI 协议)
	bash scripts/start_vllm_server.sh

vllm-server-stop:  ## 停止 vLLM Docker server
	bash scripts/stop_vllm_server.sh

vllm-server-status:  ## 检查 vLLM server 状态
	@curl -s --max-time 2 http://localhost:8000/v1/models 2>&1 | head -3 || echo "vLLM server 未运行"
