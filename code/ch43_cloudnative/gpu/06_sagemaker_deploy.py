# ---
# chapter: 43
# topic: 云原生部署与模型网关
# topic_id: cloudnative.sagemaker_deploy
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: boto3, sagemaker
# run: python 06_sagemaker_deploy.py
# expected_runtime: <1s for the default dry-run; real deployment depends on AWS
# expected_output: dry-run plan by default; real endpoint only after all explicit cost gates
# ---
# See: ../../../43_云原生部署与模型网关.md
# Interview hooks:
#   1. SageMaker SDK 与直接 EC2 + Docker 部署相比，代价/收益比？
#   2. HuggingFaceModel 与自定义 vLLM 容器的运行时边界是什么？
#   3. container_startup_health_check_timeout 应如何用真实启动数据确定？
"""
SageMaker Hugging Face endpoint 的安全部署骨架。

默认只打印计划，不导入 AWS SDK、不读取凭证、不联网。真实部署同时要求：

1. ``--deploy``；
2. 环境变量 ``SAGEMAKER_DEPLOY=1``；
3. ``--confirm-deploy CREATE_PAID_ENDPOINT``；
4. 显式选择 ``--delete-after-create`` 或 ``--keep-endpoint``。

模型、DLC 版本、实例、区域、IAM role 与 endpoint name 都必须显式提供。SageMaker
支持的 DLC 组合和模型许可证会变化，部署前须按当前官方文档再次核验。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared.gpu_guard import skip_if_mock

CONFIRM_PHRASE = "CREATE_PAID_ENDPOINT"


@dataclass(frozen=True)
class DeploymentConfig:
    model_id: str
    role_arn: str
    region: str
    endpoint_name: str
    instance_type: str
    initial_instance_count: int
    num_gpus: int
    transformers_version: str
    pytorch_version: str
    py_version: str
    startup_timeout_seconds: int
    max_input_length: int
    max_total_tokens: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe-by-default SageMaker deployment scaffold")
    parser.add_argument("--deploy", action="store_true", help="请求进入真实部署路径")
    parser.add_argument("--confirm-deploy", default="", metavar=CONFIRM_PHRASE)
    parser.add_argument("--model-id", default=os.environ.get("SAGEMAKER_MODEL_ID", ""))
    parser.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_ROLE_ARN", ""))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", ""))
    parser.add_argument("--endpoint-name", default=os.environ.get("SAGEMAKER_ENDPOINT_NAME", ""))
    parser.add_argument("--instance-type", default=os.environ.get("SAGEMAKER_INSTANCE_TYPE", ""))
    parser.add_argument("--initial-instance-count", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--transformers-version", default="")
    parser.add_argument("--pytorch-version", default="")
    parser.add_argument("--py-version", default="")
    parser.add_argument("--startup-timeout-seconds", type=int, default=600)
    parser.add_argument("--max-input-length", type=int, default=32768)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument(
        "--delete-after-create",
        action="store_true",
        help="endpoint 创建成功后立即删除，用于生命周期门禁演练",
    )
    lifecycle.add_argument(
        "--keep-endpoint",
        action="store_true",
        help="明确保留持续计费的 endpoint；完成后必须手动删除",
    )
    return parser


def _config(args: argparse.Namespace) -> DeploymentConfig:
    return DeploymentConfig(
        model_id=args.model_id.strip(),
        role_arn=args.role_arn.strip(),
        region=args.region.strip(),
        endpoint_name=args.endpoint_name.strip(),
        instance_type=args.instance_type.strip(),
        initial_instance_count=args.initial_instance_count,
        num_gpus=args.num_gpus,
        transformers_version=args.transformers_version.strip(),
        pytorch_version=args.pytorch_version.strip(),
        py_version=args.py_version.strip(),
        startup_timeout_seconds=args.startup_timeout_seconds,
        max_input_length=args.max_input_length,
        max_total_tokens=args.max_total_tokens,
    )


def _public_plan(config: DeploymentConfig, args: argparse.Namespace) -> dict[str, Any]:
    plan = asdict(config)
    for key, value in tuple(plan.items()):
        if value == "":
            plan[key] = "<required for --deploy>"
    plan.update(
        {
            "mode": "REAL DEPLOY" if args.deploy else "DRY RUN ONLY",
            "lifecycle": (
                "delete-after-create"
                if args.delete_after_create
                else ("keep-endpoint" if args.keep_endpoint else "<required for --deploy>")
            ),
            "cloud_side_effect": bool(args.deploy),
        }
    )
    return plan


def _deployment_errors(config: DeploymentConfig, args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if os.environ.get("SAGEMAKER_DEPLOY") != "1":
        errors.append("需要精确设置 SAGEMAKER_DEPLOY=1")
    if args.confirm_deploy != CONFIRM_PHRASE:
        errors.append(f"需要 --confirm-deploy {CONFIRM_PHRASE}")
    if not (args.delete_after_create or args.keep_endpoint):
        errors.append("需要显式选择 --delete-after-create 或 --keep-endpoint")

    required = {
        "--model-id": config.model_id,
        "--role-arn": config.role_arn,
        "--region": config.region,
        "--endpoint-name": config.endpoint_name,
        "--instance-type": config.instance_type,
        "--transformers-version": config.transformers_version,
        "--pytorch-version": config.pytorch_version,
        "--py-version": config.py_version,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        errors.append(f"缺少真实部署参数: {', '.join(missing)}")

    if config.role_arn and not re.fullmatch(
        r"arn:(aws|aws-cn|aws-us-gov):iam::\d{12}:role/.+",
        config.role_arn,
    ):
        errors.append("--role-arn 不是可识别的 IAM role ARN")
    if config.endpoint_name and not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", config.endpoint_name):
        errors.append("--endpoint-name 必须是 1..63 位字母、数字或连字符")
    if config.initial_instance_count < 1:
        errors.append("--initial-instance-count 必须 >= 1")
    if config.num_gpus < 1:
        errors.append("--num-gpus 必须 >= 1")
    if config.startup_timeout_seconds < 60:
        errors.append("--startup-timeout-seconds 必须 >= 60")
    if config.max_input_length < 1 or config.max_total_tokens < 1:
        errors.append("token 长度必须为正数")
    return errors


def _load_aws_runtime() -> tuple[Any, Any, Any]:
    try:
        import boto3
        import sagemaker
        from sagemaker.huggingface import HuggingFaceModel
    except ImportError as exc:
        raise RuntimeError("缺少 boto3/sagemaker；真实部署依赖必须显式安装。") from exc
    return boto3, sagemaker, HuggingFaceModel


def deploy(config: DeploymentConfig, *, delete_after_create: bool) -> str:
    """执行已通过门禁的真实部署；异常直接向上传播。"""
    boto3, sagemaker, huggingface_model_class = _load_aws_runtime()
    boto_session = boto3.Session(region_name=config.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    model = huggingface_model_class(
        env={
            "HF_MODEL_ID": config.model_id,
            "HF_TASK": "text-generation",
            "SM_NUM_GPUS": str(config.num_gpus),
            "MAX_INPUT_LENGTH": str(config.max_input_length),
            "MAX_TOTAL_TOKENS": str(config.max_total_tokens),
        },
        role=config.role_arn,
        transformers_version=config.transformers_version,
        pytorch_version=config.pytorch_version,
        py_version=config.py_version,
        sagemaker_session=sagemaker_session,
    )
    predictor = model.deploy(
        initial_instance_count=config.initial_instance_count,
        instance_type=config.instance_type,
        endpoint_name=config.endpoint_name,
        container_startup_health_check_timeout=config.startup_timeout_seconds,
    )

    if delete_after_create:
        predictor.delete_endpoint(delete_endpoint_config=True)
        print(f"[CLEANUP] endpoint 与 endpoint config 已请求删除: {config.endpoint_name}")
        return "deleted"

    print(f"[BILLING ACTIVE] endpoint 已创建并保留: {config.endpoint_name}")
    print("完成实验后立即执行：")
    print(
        f"  aws sagemaker delete-endpoint --region {config.region} "
        f"--endpoint-name {config.endpoint_name}"
    )
    print("并在 SageMaker 控制台核对 endpoint config / model 等残留资源。")
    return "active"


def main() -> int:
    if skip_if_mock("explicit SageMaker cost gates, AWS credentials, IAM permission, and a validated DLC"):
        return 0

    args = _parser().parse_args()
    config = _config(args)
    print(json.dumps(_public_plan(config, args), ensure_ascii=False, indent=2))

    if not args.deploy:
        print("DRY RUN ONLY: 未导入 AWS SDK、未读取凭证、未发请求、未创建资源。")
        print("OK")
        return 0

    errors = _deployment_errors(config, args)
    if errors:
        for error in errors:
            print(f"[REFUSE] {error}", file=sys.stderr)
        return 2

    try:
        status = deploy(config, delete_after_create=args.delete_after_create)
    except Exception as exc:
        print(
            f"[ERROR] SageMaker deploy failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "[ACTION] 到 SageMaker 控制台按 endpoint name 检查并清理可能已创建的部分资源。",
            file=sys.stderr,
        )
        return 1

    print(f"OK: deployment lifecycle status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
