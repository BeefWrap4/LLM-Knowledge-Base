# ---
# chapter: 24
# topic: 云原生部署与工程化
# section: 24.7.1 公有云 AI 服务对比 / AWS SageMaker 部署
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: sagemaker (>=2.0)
# run: python 06_sagemaker_deploy.py
# expected_runtime: 1-3s (mock) or 5-15min (real deploy)
# expected_output: prints the configured model / deploy config, with mock fallback
# ---
# See: ../tutorial/24_云原生部署与工程化.md §24.7.1
# Interview hooks:
#   1. SageMaker SDK 与直接 EC2 + Docker 部署相比，代价/收益比？
#   2. HuggingFaceModel 镜像里运行 vLLM 的话与 sagemaker-huggingface-inference-toolkit 的差异？
#   3. container_startup_health_check_timeout=600 的意义？为什么默认 60s 不够？
"""
使用 SageMaker SDK 部署大模型推理端点
"""

# Mock 模式兼容：当 sagemaker 未安装时使用 mock
try:
    import sagemaker
    from sagemaker.huggingface import HuggingFaceModel

    HAS_SAGEMAKER = True
except ImportError:
    HAS_SAGEMAKER = False

    class _MockRole:
        def __init__(self, name="MockRole"):
            self.name = name

        def __repr__(self):
            return f"<MockRole {self.name}>"

    class _MockPredictor:
        def __init__(self, endpoint_name, instance_type):
            self.endpoint_name = endpoint_name
            self.instance_type = instance_type

        def predict(self, data):
            return {"generated_text": "[MOCK prediction]", "input": data}

    class _MockHuggingFaceModel:
        def __init__(self, env, role, transformers_version, pytorch_version, py_version, model_data):
            self.env = env
            self.role = role
            self.transformers_version = transformers_version
            self.pytorch_version = pytorch_version
            self.py_version = py_version
            self.model_data = model_data
            print(f"[MockHFM] initialized: env={env}")

        def deploy(self, initial_instance_count, instance_type, container_startup_health_check_timeout):
            print(
                f"[MockHFM] deploy: {initial_instance_count}x {instance_type}, "
                f"startup_health_check_timeout={container_startup_health_check_timeout}s"
            )
            return _MockPredictor(
                endpoint_name=f"mock-endpoint-{id(self)}",
                instance_type=instance_type,
            )

    class _MockSagemaker:
        def get_execution_role(self):
            return _MockRole("arn:aws:iam::000000000000:role/MockRole")

        HuggingFaceModel = _MockHuggingFaceModel

    sagemaker = _MockSagemaker()
    HuggingFaceModel = _MockHuggingFaceModel
    print("[WARN] sagemaker not installed, using mock SDK (no real deploy)")


# 1. 创建 HuggingFace Model
hub = {
    "HF_MODEL_ID": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "HF_TASK": "text-generation",
    "SM_NUM_GPUS": "4",
    "MAX_INPUT_LENGTH": "32768",
    "MAX_TOTAL_TOKENS": "4096",
}

huggingface_model = HuggingFaceModel(
    env=hub,
    role=sagemaker.get_execution_role(),
    transformers_version="4.46",
    pytorch_version="2.5",
    py_version="py311",
    model_data=None,  # 从 HuggingFace Hub 直接加载
)

# 2. 部署到 GPU 端点
if HAS_SAGEMAKER:
    predictor = huggingface_model.deploy(
        initial_instance_count=2,  # 2 个 GPU 实例
        instance_type="ml.p4d.24xlarge",  # A100 x8
        container_startup_health_check_timeout=600,  # 模型加载需要时间
    )
else:
    predictor = huggingface_model.deploy(
        initial_instance_count=2,
        instance_type="ml.p4d.24xlarge",
        container_startup_health_check_timeout=600,
    )


# 3. 推理示例
if __name__ == "__main__":
    print("\n=== SageMaker Deployment Configuration ===")
    print(f"Model ID: {hub['HF_MODEL_ID']}")
    print(f"Task: {hub['HF_TASK']}")
    print(f"GPUs per instance: {hub['SM_NUM_GPUS']}")
    print(f"Max input length: {hub['MAX_INPUT_LENGTH']}")
    print("Instance type: ml.p4d.24xlarge (A100 x8)")
    print("Initial instance count: 2")
    print("Startup health check timeout: 600s")
    print()
    print("To call the endpoint:")
    print('  predictor.predict({"inputs": "Hello, how are you?"})')
    print()
