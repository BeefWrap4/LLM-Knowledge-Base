# ---
# chapter: 28
# topic: Secure Minions 端云协作隐私推理 (真实 mTLS 模拟)
# section: 28.6 Secure Minions (隐私推理)
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: (Python stdlib: ssl, hashlib, secrets) — 无第三方依赖
# run: python 10_secure_minions_protocol.py
# expected_runtime: <1s
# expected_output: 真实 mTLS 握手演示 (Python stdlib ssl 模块) + 端云分工
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.6, § 28.7
# Interview hooks:
#   1. Secure Minions 如何防止云端从嵌入向量重建原始数据?
#   2. 加密投影矩阵 P 的关键性质是什么 (随机正交/不可逆)?
#   3. Secure Minions 相比纯端侧方案在性能和质量上如何权衡?
"""Secure Minions 端云协作隐私推理 — 真实 Python mTLS 协议模拟.

本文件用 Python stdlib (ssl / hashlib / secrets / socket) 模拟完整
端云 mTLS 握手 + 加密通信流程, 不做真实 LLM 推理.

工作流:
  1. 端侧生成 ephemeral session key (secrets 模块, 密码学安全 RNG)
  2. 端云 mTLS 握手 (ssl 模块, TLSv1.3 + AES-256-GCM)
     - 端侧验证云端 CA 证书
     - 云端验证端侧设备证书 (双向认证)
  3. 端侧对 token IDs 做 SHA-256 (隐私: 云端看不到原始 token)
  4. 仅传 SHA-256 + 加密 session 请求
  5. 云端 LLM 推理后, 返回加密的 response
  6. 端侧解密, 仅持有 response tokens (无模型权重)

核心安全性质:
  - 端侧: 永不暴露原始 prompt
  - 云端: 永不暴露模型权重
  - 通信: mTLS 加密 + 双向认证
"""

from __future__ import annotations

import hashlib
import secrets
import socket
import ssl
import sys
import tempfile
import threading
import time
from pathlib import Path

# 让脚本既能 `python file.py` 也能 `import` 找到 shared/
_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

from shared._error_helper import raise_with_help  # noqa: E402


# ============================================================
# 1. 密码学原语 (端侧 / 云端共享)
# ============================================================
def generate_minion_session_key() -> bytes:
    """端侧: 生成临时 session key (secrets 模块, 密码学安全 RNG).

    真实场景中, 这是 ECDHE (Elliptic Curve Diffie-Hellman Ephemeral)
    的输出. 简化版用 secrets.token_bytes(32) 模拟 256-bit 共享密钥.
    """
    return secrets.token_bytes(32)  # 32 bytes = 256 bits


def hash_tokens_for_transmission(tokens: list[int]) -> str:
    """端侧: 对 token IDs 做 SHA-256, 仅传哈希 (保护原始 token).

    真实 Secure Minions 协议中, 这步是嵌入 + 随机投影, 但 demo
    简化为哈希 (更直观展示 "云端看不到原始 token").
    """
    h = hashlib.sha256()
    for t in tokens:
        h.update(t.to_bytes(4, "little"))
    return h.hexdigest()


# ============================================================
# 2. 自签名 mTLS 证书生成 (in-memory, 不入 git)
# ============================================================
def _generate_self_signed_cert(
    common_name: str,
    cert_path: Path,
    key_path: Path,
) -> None:
    """用 openssl 命令生成自签名证书 (CN=common_name).

    真实生产环境用 CA 签名证书. Demo 用自签名简化.
    若 openssl 不可用, 抛友好错.
    """
    import shutil
    import subprocess

    openssl = shutil.which("openssl")
    if not openssl:
        raise_with_help(
            "openssl 未装, 无法生成 mTLS 自签名证书.",
            "运行 `winget install OpenSSL.Beta` (Windows) 或 "
            "`brew install openssl` (Mac) / `apt install openssl` (Linux).",
        )

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 生成私钥 (RSA 2048)
    subprocess.run(
        [openssl, "genrsa", "-out", str(key_path), "2048"],
        check=True,
        capture_output=True,
        text=True,
    )
    # 2) 生成自签名证书 (有效期 1 天, demo 足够)
    subprocess.run(
        [
            openssl,
            "req",
            "-new",
            "-x509",
            "-key",
            str(key_path),
            "-out",
            str(cert_path),
            "-days",
            "1",
            "-subj",
            f"/CN={common_name}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


# ============================================================
# 3. mTLS 握手 (单进程 server + client, 用临时端口)
# ============================================================
def simulate_mtls_handshake(
    server_cn: str = "llm-server",
    client_cn: str = "minion-client",
    timeout_s: float = 5.0,
) -> dict:
    """模拟端云 mTLS 握手 (单进程, 临时 socket).

    工作流:
      1. 生成自签名 server.crt + client.crt (共用同一 CA 简化)
      2. 启动子线程做 TLS server (云端)
      3. 主线程做 TLS client (端侧)
      4. 互相验证证书 → 握手成功 → 加密通信 1 个 RTT

    Returns:
        dict 含握手结果 (cipher, protocol, peer cert CN, 延迟等)
    """
    tmpdir = Path(tempfile.gettempdir()) / "secure_minions_demo"
    tmpdir.mkdir(parents=True, exist_ok=True)
    server_cert = tmpdir / "server.crt"
    server_key = tmpdir / "server.key"
    client_cert = tmpdir / "client.crt"
    client_key = tmpdir / "client.key"
    ca_cert = tmpdir / "ca.crt"

    # 简化: 用同一份证书当 CA (demo only, 真实场景分开)
    _generate_self_signed_cert(server_cn, server_cert, server_key)
    _generate_self_signed_cert(client_cn, client_cert, client_key)
    ca_cert.write_bytes(server_cert.read_bytes())  # server 自签 = "CA"

    # ---- 启动 server 线程 ----
    server_result: dict = {}
    server_ready = threading.Event()

    def tls_server_thread(port: int) -> None:
        """云端 mTLS server: 验证客户端证书."""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.load_cert_chain(certfile=str(server_cert), keyfile=str(server_key))
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cafile=str(ca_cert))
        # 由于 client cert 是另一份自签, 单独验
        ctx.load_verify_locations(cafile=str(client_cert))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as raw_sock:
            raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            raw_sock.bind(("127.0.0.1", port))
            raw_sock.listen(1)
            server_ready.set()
            try:
                conn, _ = raw_sock.accept()
            except TimeoutError:
                server_result["error"] = "accept timeout"
                return
            with conn:
                try:
                    tls_conn = ctx.wrap_socket(conn, server_side=True)
                    # 互相读 1 字节确认握手完成
                    data = tls_conn.recv(64)
                    server_result.update(
                        {
                            "cipher": tls_conn.cipher()[0] if tls_conn.cipher() else "?",
                            "protocol": tls_conn.version(),
                            "peer_cn": "?",
                            "received": data.decode("utf-8", errors="replace"),
                        }
                    )
                    # 读 peer cert CN
                    peer_cert = tls_conn.getpeercert(binary_form=False)
                    if peer_cert:
                        for tup in peer_cert.get("subject", []):
                            for k, v in tup:
                                if k == "commonName":
                                    server_result["peer_cn"] = v
                    tls_conn.sendall(b"ok")
                except ssl.SSLError as e:
                    server_result["error"] = f"SSL error: {e}"

    # 找空闲端口
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    t = threading.Thread(target=tls_server_thread, args=(port,), daemon=True)
    t.start()
    if not server_ready.wait(timeout=2.0):
        raise_with_help(
            "mTLS server 启动失败.",
            "检查端口绑定权限 / 防火墙.",
        )

    # ---- 客户端连接 (端侧) ----
    client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client_ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    client_ctx.load_cert_chain(certfile=str(client_cert), keyfile=str(client_key))
    client_ctx.verify_mode = ssl.CERT_REQUIRED
    client_ctx.load_verify_locations(cafile=str(ca_cert))
    client_ctx.load_verify_locations(cafile=str(server_cert))

    start = time.perf_counter()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as raw_sock:
            with client_ctx.wrap_socket(raw_sock, server_hostname=server_cn) as tls_sock:
                # 验证 server 证书
                server_cert_peer = tls_sock.getpeercert(binary_form=False)
                server_cn_peer = "?"
                if server_cert_peer:
                    for tup in server_cert_peer.get("subject", []):
                        for k, v in tup:
                            if k == "commonName":
                                server_cn_peer = v
                # 发 1 字节 + 收 1 字节
                tls_sock.sendall(b"hi")
                ack = tls_sock.recv(64)
                handshake_ms = (time.perf_counter() - start) * 1000
                client_cipher = tls_sock.cipher()[0] if tls_sock.cipher() else "?"
                client_protocol = tls_sock.version()
    except (ssl.SSLError, OSError) as e:
        raise_with_help(
            f"mTLS 客户端连接失败: {e}",
            "检查 openssl 是否正确生成证书 / 端口是否被占用.",
        )

    t.join(timeout=2.0)

    return {
        "client_cipher": client_cipher,
        "client_protocol": client_protocol,
        "client_server_cn": server_cn_peer,
        "client_ack": ack.decode("utf-8", errors="replace"),
        "server_cipher": server_result.get("cipher", "?"),
        "server_protocol": server_result.get("protocol", "?"),
        "server_peer_cn": server_result.get("peer_cn", "?"),
        "server_received": server_result.get("received", "?"),
        "handshake_ms": round(handshake_ms, 2),
        "cert_dir": str(tmpdir),
    }


# ============================================================
# 4. 端云协议完整演示
# ============================================================
def run_secure_minions_demo() -> dict:
    """完整端云 mTLS 协议演示 (无第三方依赖)."""
    print("=" * 60)
    print("Secure Minions 端云协作隐私推理 — 真实 mTLS 模拟")
    print("=" * 60)

    # 1) 端侧生成 session key
    session_key = generate_minion_session_key()
    print("\n[1] 端侧生成 ephemeral session key (256-bit)")
    print(f"    key preview: {session_key[:8].hex()}...{session_key[-4:].hex()}")

    # 2) 端云 mTLS 握手
    print("\n[2] 端云 mTLS 握手 (TLSv1.3 + AES-256-GCM)...")
    handshake = simulate_mtls_handshake()
    print(f"    client → server  cipher: {handshake['client_cipher']}")
    print(f"    client ← server  protocol: {handshake['client_protocol']}")
    print(f"    端侧验证 云端证书 CN: {handshake['client_server_cn']}")
    print(f"    云端验证 端侧证书 CN: {handshake['server_peer_cn']}")
    print(f"    握手耗时: {handshake['handshake_ms']}ms")
    print(f"    临时证书目录: {handshake['cert_dir']}")

    # 3) 端侧对 token IDs 做 SHA-256
    mock_tokens = [9906, 1917, 3186]  # 模拟 tokenizer.encode("Hello world!")
    print("\n[3] 端侧对 token IDs 做 SHA-256 (隐私保护)")
    print(f"    原始 tokens: {mock_tokens}  (永不离开端侧)")
    token_hash = hash_tokens_for_transmission(mock_tokens)
    print(f"    token SHA-256: {token_hash[:32]}...{token_hash[-8:]}")

    # 4) 加密 session 请求 (模拟)
    request_fingerprint = hashlib.sha256(session_key + token_hash.encode()).hexdigest()
    print("\n[4] 加密 session 请求 (TLS channel 保护)")
    print(f"    request fingerprint: {request_fingerprint[:32]}...{request_fingerprint[-8:]}")

    # 5) 云端 LLM 推理 (mock) + 返回加密 response
    print("\n[5] 云端 LLM 推理 (mock) + 加密 response")
    mock_response_tokens = [13, 5782, 318, 1128, 13, 0]  # 模拟 "<|im_start|>..."
    response_hash = hash_tokens_for_transmission(mock_response_tokens)
    print(f"    response tokens 数量: {len(mock_response_tokens)}")
    print(f"    response SHA-256: {response_hash[:32]}...{response_hash[-8:]}")

    # 6) 端侧解密
    print("\n[6] 端侧解密 response (TLS channel 保护)")
    print(f"    端侧收到: {len(mock_response_tokens)} 个 token (无模型权重)")

    return {
        "session_key_bytes": len(session_key),
        "handshake_ms": handshake["handshake_ms"],
        "token_count": len(mock_tokens),
        "token_sha256": token_hash[:16] + "...",
        "request_fingerprint": request_fingerprint[:16] + "...",
        "response_count": len(mock_response_tokens),
        "cipher": handshake["client_cipher"],
        "protocol": handshake["client_protocol"],
    }


# ============================================================
# 5. 主流程
# ============================================================
def main() -> None:
    print("=== Secure Minions Protocol (mTLS) ===\n")
    print("场景: 端云分离 LLM 推理")
    print("  端侧: 用户 prompt, 临时 session key")
    print("  云端: LLM 模型权重")
    print("  协议: mTLS 加密 token 哈希\n")

    result = run_secure_minions_demo()
    print()
    print("--- 协议结果汇总 ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()
    print("✅ 端侧: 永不暴露原始 prompt (仅传 SHA-256)")
    print("✅ 云端: 永不暴露模型权重 (仅返回 token 哈希)")
    print("✅ 通信: mTLS 加密 + 双向证书认证")


if __name__ == "__main__":
    main()
