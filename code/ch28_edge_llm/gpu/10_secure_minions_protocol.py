# ---
# chapter: 28
# topic: Secure Minions 安全边界 (TLS 教学演示，不含 TEE)
# section: 28.6 Secure Minions (机密计算)
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: (Python stdlib: ssl, hashlib, secrets) — 无第三方依赖
# run: python 10_secure_minions_protocol.py
# expected_runtime: <1s
# expected_output: 本地 mTLS 握手演示 + TLS/TEE 安全边界说明
# ---
# See: ../tutorial/28_端侧与边缘LLM.md § 28.6, § 28.7
# Interview hooks:
#   1. TLS、远程证明和 TEE 分别解决什么问题?
#   2. 为什么哈希 prompt/token 不能替代机密推理?
#   3. Secure Minions 相比纯端侧方案扩大了哪些信任边界?
"""Secure Minions 安全边界教学：本地 mTLS 演示，不是官方协议实现。

本文件只用 Python stdlib (ssl / hashlib / secrets / socket) 演示
双向 TLS 的传输加密与身份认证，不做真实 LLM 推理，也不实现
Secure Minions Secure Chat 所需的远程证明、机密 CPU/GPU 或 TEE。

工作流:
  1. 端侧生成一次性请求 nonce（仅用于演示请求关联）
  2. 端云 mTLS 握手 (ssl 模块, TLSv1.3)
     - 端侧验证云端 CA 证书
     - 云端验证端侧设备证书 (双向认证)
  3. 用 SHA-256 生成日志指纹（仅用于完整性/关联，不提供 prompt 隐私）
  4. 明确真实推理服务必须在 TEE 内看到明文 prompt 才能推理
  5. 云端 mock 推理后，通过 TLS 返回明文响应

安全边界:
  - 本演示能证明：通信链路加密 + 双向证书认证
  - 本演示不能证明：云端运营方看不到 prompt、代码/模型在可信 TEE 中运行
  - 生产 Secure Minions 还需要远程证明与机密计算基础设施
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
from shared.gpu_guard import skip_if_mock  # noqa: E402


# ============================================================
# 1. 演示用 nonce 与日志指纹
# ============================================================
def generate_request_nonce() -> bytes:
    """生成一次性请求 nonce；TLS 会自行协商会话密钥。"""
    return secrets.token_bytes(32)


def fingerprint_tokens(tokens: list[int]) -> str:
    """为日志/测试生成确定性指纹，不把哈希误当作隐私保护。

    Token 空间小且结构化，哈希可能被字典攻击；模型也无法仅凭哈希
    执行常规推理。实际请求仍会在 TLS 连接内传输，并在受信 TEE 内解密。
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
    config_path = cert_path.with_suffix(".cnf")
    config_path.write_text(
        "\n".join(
            [
                "[req]",
                "distinguished_name = subject",
                "prompt = no",
                "x509_extensions = certificate_extensions",
                "",
                "[subject]",
                f"CN = {common_name}",
                "",
                "[certificate_extensions]",
                "basicConstraints = critical,CA:TRUE",
                "keyUsage = critical,digitalSignature,keyEncipherment,keyCertSign",
                "extendedKeyUsage = serverAuth,clientAuth",
                f"subjectAltName = DNS:{common_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # 1) 生成私钥 (RSA 2048)
    subprocess.run(
        [openssl, "genrsa", "-out", str(key_path), "2048"],
        check=True,
        capture_output=True,
        text=True,
    )
    # 2) 生成自签名证书 (有效期 1 天, demo 足够)
    try:
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
                "-config",
                str(config_path),
                "-extensions",
                "certificate_extensions",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise_with_help(
            f"openssl 生成演示证书失败: {detail}",
            f"检查 OpenSSL 可执行文件 `{openssl}`；本示例不会修改系统 OpenSSL 配置。",
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
    """运行 TLS 教学演示，并显式报告未实现的 Secure Minions 能力。"""
    print("=" * 60)
    print("Secure Minions 安全边界 — 本地 mTLS 教学演示")
    print("=" * 60)

    # 1) 端侧生成请求 nonce；TLS 会自行协商真正的会话密钥
    request_nonce = generate_request_nonce()
    print("\n[1] 端侧生成一次性请求 nonce (256-bit)")
    print(f"    nonce preview: {request_nonce[:8].hex()}...{request_nonce[-4:].hex()}")

    # 2) 端云 mTLS 握手
    print("\n[2] 端云 mTLS 握手 (TLSv1.3 + AES-256-GCM)...")
    handshake = simulate_mtls_handshake()
    print(f"    client → server  cipher: {handshake['client_cipher']}")
    print(f"    client ← server  protocol: {handshake['client_protocol']}")
    print(f"    端侧验证 云端证书 CN: {handshake['client_server_cn']}")
    print(f"    云端验证 端侧证书 CN: {handshake['server_peer_cn']}")
    print(f"    握手耗时: {handshake['handshake_ms']}ms")
    print(f"    临时证书目录: {handshake['cert_dir']}")

    # 3) SHA-256 只作为日志指纹，绝不声称它能隐藏低熵 token
    mock_tokens = [9906, 1917, 3186]  # 模拟 tokenizer.encode("Hello world!")
    print("\n[3] 生成请求日志指纹（不是隐私机制）")
    print(f"    原始 tokens: {mock_tokens}")
    token_fingerprint = fingerprint_tokens(mock_tokens)
    print(f"    token fingerprint: {token_fingerprint[:32]}...{token_fingerprint[-8:]}")
    print("    注意: 服务端必须在可信执行环境内看到明文，才能执行常规 LLM 推理")

    # 4) TLS 保护传输；指纹只用于关联请求
    request_fingerprint = hashlib.sha256(request_nonce + token_fingerprint.encode()).hexdigest()
    print("\n[4] 通过 TLS 通道传输请求")
    print(f"    request fingerprint: {request_fingerprint[:32]}...{request_fingerprint[-8:]}")

    # 5) 云端 mock 推理 + TLS 返回响应
    print("\n[5] 云端 mock 推理 + TLS 返回响应")
    mock_response_tokens = [13, 5782, 318, 1128, 13, 0]  # 模拟 "<|im_start|>..."
    response_fingerprint = fingerprint_tokens(mock_response_tokens)
    print(f"    response tokens 数量: {len(mock_response_tokens)}")
    print(f"    response fingerprint: {response_fingerprint[:32]}...{response_fingerprint[-8:]}")

    # 6) 报告本演示的能力边界
    print("\n[6] 能力边界")
    print("    已演示: TLS 传输加密、双向证书认证")
    print("    未实现: 远程证明、机密 CPU/GPU、TEE 内推理")

    return {
        "request_nonce_bytes": len(request_nonce),
        "handshake_ms": handshake["handshake_ms"],
        "token_count": len(mock_tokens),
        "token_fingerprint": token_fingerprint[:16] + "...",
        "request_fingerprint": request_fingerprint[:16] + "...",
        "response_count": len(mock_response_tokens),
        "cipher": handshake["client_cipher"],
        "protocol": handshake["client_protocol"],
        "remote_attestation": "NOT_IMPLEMENTED",
        "confidential_compute": "NOT_IMPLEMENTED",
    }


# ============================================================
# 5. 主流程
# ============================================================
def main() -> None:
    if skip_if_mock("OpenSSL, local loopback sockets, and temporary certificate files"):
        return
    print("=== Secure Minions Security Boundary (TLS-only demo) ===\n")
    print("场景: 端云分离 LLM 推理的传输层教学")
    print("  本地: mTLS 握手与请求/响应指纹")
    print("  缺失: 远程证明、TEE 与机密 GPU")
    print("  结论: 这不是 Secure Minions 官方协议或生产实现\n")

    result = run_secure_minions_demo()
    print()
    print("--- 协议结果汇总 ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()
    print("✅ 通信链路: mTLS 加密 + 双向证书认证")
    print("⚠️  服务端可信性: 本示例没有远程证明或 TEE，不能防云端运营方读取明文")
    print("⚠️  SHA-256: 仅作指纹，不是 prompt 隐私或 LLM 推理协议")
    print("OK")


if __name__ == "__main__":
    main()
