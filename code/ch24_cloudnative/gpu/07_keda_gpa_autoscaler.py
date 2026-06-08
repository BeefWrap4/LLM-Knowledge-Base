# ---
# chapter: 24
# topic: 云原生部署与工程化
# section: 24.7.3 成本优化策略 / KEDA ScaledObject
# difficulty: ⭐⭐⭐⭐
# tier: gpu
# deps: (stdlib only; YAML as embedded string)
# run: python 07_keda_gpa_autoscaler.py
# expected_runtime: <1s
# expected_output: prints parsed KEDA ScaledObject YAML, lists triggers, "OK"
# ---
# See: ../tutorial/24_云原生部署与工程化.md §24.7.3
# Interview hooks:
#   1. KEDA 与 K8s HPA 的关系？为什么 HPA 不够用，需要 KEDA？
#   2. cooldownPeriod: 300 在 GPU 推理中的意义？过短会出什么问题？
#   3. 基于 Prometheus 指标做弹性时，如何避免查询过载？
"""
基于 Prometheus 指标的 GPU 节点自动缩容策略
结合 K8s HPA + KEDA 实现事件驱动弹性

本脚本演示 KEDA ScaledObject 配置的 Python 表示与可读性检查。
"""

# KEDA ScaledObject 配置示例（基于 GPU 指标的事件驱动弹性）
KEDA_SCALED_OBJECT = """
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-inference-scaler
  namespace: llm-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference-server
  minReplicaCount: 1
  maxReplicaCount: 8
  cooldownPeriod: 300   # 缩容冷却 5 分钟
  triggers:
    # 🆕 2026: 基于 GPU 利用率的触发
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: DCGM_FI_DEV_GPU_UTIL
        threshold: "75"       # GPU 利用率 > 75% 触发扩容
        query: |
          avg(
            DCGM_FI_DEV_GPU_UTIL{
              namespace="llm-inference",
              deployment="llm-inference-server"
            }
          )
    # 推理队列深度触发
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: llm_inference_queue_depth
        threshold: "20"
        query: |
          sum(llm_inference_queue_depth{
            namespace="llm-inference"
          })
"""


def parse_keda_yaml(yaml_text: str) -> dict:
    """轻量 YAML 解析（不依赖 PyYAML 的情况下用正则提取关键字段）。"""
    import re

    result = {
        "apiVersion": None,
        "kind": None,
        "name": None,
        "namespace": None,
        "scaleTarget": {},
        "minReplicaCount": None,
        "maxReplicaCount": None,
        "cooldownPeriod": None,
        "triggers": [],
    }

    result["apiVersion"] = re.search(r"apiVersion:\s*(\S+)", yaml_text).group(1)
    result["kind"] = re.search(r"kind:\s*(\S+)", yaml_text).group(1)
    result["name"] = re.search(r"name:\s*(\S+)", yaml_text).group(1)
    result["namespace"] = re.search(r"namespace:\s*(\S+)", yaml_text).group(1)
    result["minReplicaCount"] = int(re.search(r"minReplicaCount:\s*(\d+)", yaml_text).group(1))
    result["maxReplicaCount"] = int(re.search(r"maxReplicaCount:\s*(\d+)", yaml_text).group(1))
    cooldown = re.search(r"cooldownPeriod:\s*(\d+)", yaml_text)
    if cooldown:
        result["cooldownPeriod"] = int(cooldown.group(1))

    # 解析 triggers
    trigger_blocks = re.findall(
        r"-\s*type:\s*(\S+)\s*\n\s*metadata:\s*\n([\s\S]*?)(?=\n\s*-\s*type:|\Z)",
        yaml_text,
    )
    for trig_type, meta_block in trigger_blocks:
        metric_name = re.search(r"metricName:\s*(\S+)", meta_block)
        threshold = re.search(r"threshold:\s*\"?(\S+?)\"?\s*(?:\n|$)", meta_block)
        server = re.search(r"serverAddress:\s*(\S+)", meta_block)
        result["triggers"].append(
            {
                "type": trig_type,
                "metricName": metric_name.group(1) if metric_name else None,
                "threshold": threshold.group(1) if threshold else None,
                "serverAddress": server.group(1) if server else None,
            }
        )

    return result


if __name__ == "__main__":
    parsed = parse_keda_yaml(KEDA_SCALED_OBJECT)
    print("=== Parsed KEDA ScaledObject ===")
    print(f"Name: {parsed['name']}")
    print(f"Namespace: {parsed['namespace']}")
    print(f"Replicas: {parsed['minReplicaCount']} - {parsed['maxReplicaCount']}")
    print(f"Cooldown: {parsed['cooldownPeriod']}s")
    print(f"Triggers: {len(parsed['triggers'])}")
    for t in parsed["triggers"]:
        print(f"  - type={t['type']}, metric={t['metricName']}, threshold={t['threshold']}")
        print(f"    server={t['serverAddress']}")

    # 校验：min < max、cooldown > 0
    assert parsed["minReplicaCount"] < parsed["maxReplicaCount"], "min must be < max"
    assert parsed["cooldownPeriod"] > 0, "cooldown must be > 0"
    print("\nValidation PASSED")
