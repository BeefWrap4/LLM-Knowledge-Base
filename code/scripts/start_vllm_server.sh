#!/usr/bin/env bash
# ---
# code/scripts/start_vllm_server.sh
# 启动 vLLM Docker 容器 (server 模式), 让本机 vLLM 例子可走 OpenAI 协议连
# Usage:
#   bash scripts/start_vllm_server.sh                    # 默认 Qwen2.5-0.5B
#   MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8001 bash scripts/start_vllm_server.sh
# ---
# Git Bash on Windows 会自动把 /root/... 转成 D:/softwares/Git/root/...
# 关掉 MSYS 路径转换以保留容器内绝对路径
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

set -e

MODEL="${MODEL:-code/models/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-server}"
GPU_MEM="${GPU_MEM:-0.5}"
MAX_LEN="${MAX_LEN:-2048}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"

# 计算绝对路径 (vLLM 容器内挂载需要绝对路径)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${CODE_ROOT}/models"

# 把 MODEL 解析成容器内路径 (HF cache 标准位置)
# 本机原始路径 code/models/X 或绝对路径 都映射到 /root/.cache/huggingface/X
case "${MODEL}" in
    /*) MODEL_ABS="${MODEL}" ;;
    *)  MODEL_ABS="${CODE_ROOT}/${MODEL}" ;;
esac
# 取模型目录名 (最后一段), 拼到容器内 HF cache 路径
MODEL_DIR_NAME="$(basename "${MODEL}")"
MODEL_IN_CONTAINER="/root/.cache/huggingface/${MODEL_DIR_NAME}"

# 检查 docker
if ! command -v docker &> /dev/null; then
    echo "[ERROR] docker 未装"; exit 1
fi

# 检查 GPU passthrough
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "[ERROR] Docker GPU passthrough 不可用, 请先装 nvidia-container-toolkit"; exit 1
fi

# 检查是否已运行
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] ${CONTAINER_NAME} 已在运行"
    echo "  端点: http://localhost:${PORT}/v1"
    exit 0
fi

# 删除旧的 stopped container
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "[启动 vLLM Docker 容器]"
echo "  镜像:    ${IMAGE}"
echo "  模型:    ${MODEL_IN_CONTAINER}  (从本机 ${MODEL_DIR_NAME}/)"
echo "  端口:    ${PORT}"
echo "  GPU mem: ${GPU_MEM}"
echo "  max len: ${MAX_LEN}"
echo

docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -p "${PORT}:8000" \
    -v "${MODELS_DIR}:/root/.cache/huggingface" \
    -e HF_HUB_OFFLINE=0 \
    "${IMAGE}" \
    --model "${MODEL_IN_CONTAINER}" \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-model-len "${MAX_LEN}" \
    --enforce-eager
# 注: Git Bash on Windows 会把 /root/... 自动转成 D:/softwares/Git/root/...
# 上面的 --model 参数已通过 MSYS_NO_PATHCONV=1 避免路径转换

echo
echo "等待 server 启动 (30-90 秒)..."
for i in $(seq 1 30); do
    if curl -s --max-time 2 "http://localhost:${PORT}/v1/models" &> /dev/null; then
        echo
        echo "vLLM server 已就绪: http://localhost:${PORT}/v1"
        echo
        echo "测试:"
        echo "  curl http://localhost:${PORT}/v1/models"
        echo "  curl -X POST http://localhost:${PORT}/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\":\"${MODEL_DIR_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":32}'"
        exit 0
    fi
    sleep 3
    echo -n "."
done

echo
echo "[ERROR] vLLM server 90 秒内未就绪, 查看日志:"
docker logs --tail 50 "${CONTAINER_NAME}"
exit 1
