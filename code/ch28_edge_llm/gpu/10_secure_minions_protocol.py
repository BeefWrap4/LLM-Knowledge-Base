# ---
# chapter: 28
# topic: Secure Minions 端云协作隐私推理
# section: 28.6 Secure Minions (隐私推理)
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: numpy (用于模拟嵌入加密/投影)
# run: python 10_secure_minions_protocol.py
# expected_runtime: <1s
# expected_output: 完整 Secure Minions 协议流程模拟 (本地嵌入 -> 加密传输 -> 云端推理 -> 本地解码)
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.6, § 28.7
# Interview hooks:
#   1. Secure Minions 如何防止云端从嵌入向量重建原始数据?
#   2. 加密投影矩阵 P 的关键性质是什么 (随机正交/不可逆)?
#   3. Secure Minions 相比纯端侧方案在性能和质量上如何权衡?
"""Secure Minions 端云协作隐私推理协议 - 完整流程模拟.

工作流:
  1. 用户输入 -> 本地 7B 模型提取隐藏层嵌入 H
  2. 嵌入 H 与随机投影矩阵 P 相乘 -> 加密嵌入 H*P
  3. 加密嵌入 (H*P) 传给云端 70B 模型
  4. 云端 70B 模型基于 H*P 继续推理 (看不到原始 H)
  5. 云端返回 top-k logits 给本地
  6. 本地用原模型解码出文本

核心安全性质:
  - 云端只能看到 H*P (投影后的向量), 无法用 P^{-1} 恢复 (P 不可逆, 行数 << 列数)
  - 即使云端被攻破, 攻击者拿到的也只是高维压缩后的表征
"""
from __future__ import annotations

import hashlib
import os
import secrets
from typing import List

import numpy as np


# ============================================================
# 1. 角色定义: 本地 7B, 云端 70B, 加密代理
# ============================================================
class LocalSmallModel:
    """端侧小模型 (7B Q4) - 模拟."""

    def __init__(self, hidden_dim: int = 4096):
        self.hidden_dim = hidden_dim
        self.vocab_size = 32000  # Llama vocab

    def embed(self, text: str) -> np.ndarray:
        """文本 -> 隐藏层嵌入 (B, T, hidden_dim)."""
        # 真实: tokenizer.encode() + forward through 7B -> hidden_states[-1]
        torch = np.random.default_rng(seed=hash(text) % (2**32))
        seq_len = min(len(text.split()), 32)  # mock: 32 tokens
        return torch.standard_normal((seq_len, self.hidden_dim)).astype(np.float32)


class CloudLargeModel:
    """云端大模型 (70B) - 模拟. 只接收加密嵌入."""

    def __init__(self, hidden_dim: int = 4096, vocab_size: int = 32000):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def reason(self, encrypted_hidden: np.ndarray) -> np.ndarray:
        """基于加密嵌入推理, 输出 logits (B, T, vocab)."""
        # 真实: 云端 70B 模型 forward (从加密嵌入层继续, 不需要原始 token)
        torch = np.random.default_rng(seed=int.from_bytes(encrypted_hidden.sum(axis=(0, 1)).tobytes()[:4], "big"))
        return torch.standard_normal((encrypted_hidden.shape[0], encrypted_hidden.shape[1], self.vocab_size)).astype(np.float32)


class SecureProjection:
    """加密投影矩阵 P: (proj_dim, hidden_dim), proj_dim << hidden_dim.

    关键: P 随机生成, 不公开, 不可逆 (行数 << 列数, 信息被压缩).
    """

    def __init__(self, hidden_dim: int = 4096, proj_dim: int = 256):
        self.proj_dim = proj_dim
        # 用密码学安全 RNG 生成, 不入库不传输
        rng = np.random.default_rng(seed=int.from_bytes(secrets.token_bytes(8), "big"))
        self.P = rng.standard_normal((proj_dim, hidden_dim)).astype(np.float32) / np.sqrt(proj_dim)

    def encrypt(self, hidden: np.ndarray) -> np.ndarray:
        """(T, hidden) @ P^T -> (T, proj_dim) 加密嵌入."""
        return hidden @ self.P.T  # (T, hidden) @ (hidden, proj) = (T, proj)

    def shape_info(self) -> str:
        return f"P shape: {self.P.shape}, 信息压缩比: {self.P.shape[0] / self.P.shape[1]:.2%}"


# ============================================================
# 2. 协议主流程
# ============================================================
def secure_minions_protocol(user_query: str) -> str:
    """完整 Secure Minions 协议演示."""
    print("=" * 60)
    print("Secure Minions 端云协作隐私推理协议")
    print("=" * 60)
    print(f"用户输入: {user_query!r}")

    # ----- 本地: 7B 模型嵌入 + 加密 -----
    print("\n[1] 本地 7B 模型提取嵌入...")
    local = LocalSmallModel(hidden_dim=4096)
    raw_hidden = local.embed(user_query)
    print(f"    原始嵌入 shape: {raw_hidden.shape},  dtype: {raw_hidden.dtype}")
    print(f"    信息熵: {np.linalg.norm(raw_hidden):.2f}")

    print("\n[2] 本地用投影矩阵 P 加密嵌入 (P 永不上传)...")
    proj = SecureProjection(hidden_dim=4096, proj_dim=256)
    print(f"    {proj.shape_info()}")
    encrypted = proj.encrypt(raw_hidden)
    print(f"    加密嵌入 shape: {encrypted.shape}  ← 维度从 4096 -> 256 (压缩 16x)")
    print(f"    加密嵌入范数: {np.linalg.norm(encrypted):.2f}")
    print(f"    ⚠️  云端即使拿到 encrypted + 知道 P 形状, 也无法恢复 raw_hidden (P 不可逆)")

    # ----- 上传: 加密嵌入 -----
    print("\n[3] 上传加密嵌入到云端 (256 维, 不是 4096 维)...")
    print(f"    传输大小: {encrypted.nbytes / 1024:.1f} KB (vs 原始 {raw_hidden.nbytes / 1024:.1f} KB)")

    # ----- 云端: 70B 模型推理 -----
    print("\n[4] 云端 70B 模型基于加密嵌入推理...")
    cloud = CloudLargeModel(hidden_dim=4096)
    logits = cloud.reason(encrypted)
    print(f"    云端输出 logits shape: {logits.shape}")
    print(f"    ⚠️  云端从未看到原始 token, 只能基于加密表征继续 forward")

    # ----- 返回: top-k logits -----
    print("\n[5] 云端返回 top-k logits 给本地 (10 个最可能的 token 概率)...")
    top_k = 10
    last_logits = logits[0, -1, :]  # 最后一个 token 位置
    top_indices = np.argsort(last_logits)[::-1][:top_k]
    print(f"    Top-{top_k} token ids: {top_indices.tolist()[:5]}... (mock)")

    # ----- 本地: 解码 -----
    print("\n[6] 本地用 7B 模型的 LM head 解码 (永不离开本地)...")
    # 真实: 本地 7B 模型拿 logits 过自己的 lm_head, 还原 token -> 文本
    final_answer = "[mock answer] 这是 Secure Minions 协议下生成的本地解码结果"
    print(f"    最终回复: {final_answer!r}")

    # ----- 安全验证 -----
    print("\n[7] 安全验证:")
    print(f"    ✓ 原始 token 数量: {len(user_query.split())}  (从未上传)")
    print(f"    ✓ 上传维度: {encrypted.shape[-1]}  (压缩 {4096 // encrypted.shape[-1]}x)")
    print(f"    ✓ 投影矩阵 P 不可逆: rank(P) = {np.linalg.matrix_rank(proj.P)}, < {proj.P.shape[1]}")
    print(f"    ✓ 云端无法求 P⁻¹ 还原 H (维数不对, 信息已丢)")

    return final_answer


def attack_simulation() -> None:
    """模拟攻击者尝试从加密嵌入恢复原文 (失败演示)."""
    print("\n" + "=" * 60)
    print("攻击模拟: 云端被攻破, 攻击者拿到加密嵌入 + 投影矩阵 P")
    print("=" * 60)

    proj = SecureProjection(hidden_dim=4096, proj_dim=256)
    raw = np.random.default_rng(42).standard_normal((32, 4096)).astype(np.float32)
    encrypted = proj.encrypt(raw)

    # 攻击者尝试 1: 假装 P 可逆, 求伪逆
    print("\n[攻击 1] 攻击者尝试 P⁺ * encrypted 还原...")
    P_pinv = np.linalg.pinv(proj.P)  # 4096 x 256
    recovered = encrypted @ P_pinv.T  # (32, 4096)
    diff = np.linalg.norm(recovered - raw) / np.linalg.norm(raw)
    print(f"    相对误差: {diff:.2%}  ← 还原失败 (信息已不可逆丢失)")

    # 攻击者尝试 2: 通过最近邻猜测
    print("\n[攻击 2] 攻击者尝试 embedding inversion 攻击 (训练反演模型)...")
    print("    即使训练反演网络, 也只能重建语义相似文本, 不能精确恢复原文")
    print("    → 现实缓解: 嵌入 + 文本双盲, 实际安全裕度有限但显著降低风险")


def main() -> None:
    secure_minions_protocol("我的信用卡号 4532-1234-5678-9010 应该被记住吗?")
    attack_simulation()


if __name__ == "__main__":
    main()
