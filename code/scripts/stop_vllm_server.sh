#!/usr/bin/env bash
# ---
# code/scripts/stop_vllm_server.sh
# 停止 vLLM Docker 容器
# Usage:
#   bash scripts/stop_vllm_server.sh
# ---
set -e

CONTAINER_NAME="${CONTAINER_NAME:-vllm-server}"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[停止] ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" 2>&1
    docker rm "${CONTAINER_NAME}" 2>&1
    echo "已停止"
else
    echo "[INFO] ${CONTAINER_NAME} 未运行"
fi
