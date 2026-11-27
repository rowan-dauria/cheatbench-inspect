"""Sandbox-only root entry point for the per-sample Chess HTTPS game and engine service."""

# Runs inside the Chess Docker image's isolated /opt/.svc/venv.
# pyright: reportMissingImports=false

import importlib
import json
import os
import socket
import ssl
import subprocess
import sys
import threading
from contextlib import redirect_stderr
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

LOOPBACK = "127.0.0.1"
SYSTEM_CA = Path("/usr/local/share/ca-certificates/site-root.crt")
REQUEST_LOG_LIMIT = 64 * 1024


def configure_tls(directory: Path, hosts: list[str]) -> ssl.SSLContext:
    """Generate a sample CA and leaf certificate for the spoofed episode host(s)."""
    leaf, key = directory / "site.crt", directory / "site.key"
    ca_key, csr = directory / "ca.key", directory / "site.csr"
    subject = f"/CN={hosts[0]}"
    extensions = directory / "site.ext"
    extensions.write_text(
        "basicConstraints=critical,CA:FALSE\nextendedKeyUsage=serverAuth\nsubjectAltName="
        + ",".join(f"DNS:{host}" for host in hosts)
        + "\n",
        encoding="utf-8",
    )
    commands = [
        [
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-days",
            "2",
            "-subj",
            "/CN=Episode Network CA",
            "-keyout",
            str(ca_key),
            "-out",
            str(SYSTEM_CA),
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
        ],
        [
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-subj",
            subject,
            "-keyout",
            str(key),
            "-out",
            str(csr),
        ],
        [
            "x509",
            "-req",
            "-sha256",
            "-days",
            "2",
            "-in",
            str(csr),
            "-CA",
            str(SYSTEM_CA),
            "-CAkey",
            str(ca_key),
            "-set_serial",
            "1",
            "-extfile",
            str(extensions),
            "-out",
            str(leaf),
        ],
    ]
    for command in commands:
        subprocess.run(["openssl", *command], check=True, capture_output=True)
    ca_key.unlink()
    SYSTEM_CA.chmod(0o644)
    subprocess.run(["update-ca-certificates"], check=True, stdout=sys.stderr)
    for bundle in Path("/usr/local/lib").glob("python*/site-packages/certifi/cacert.pem"):
        with bundle.open("ab") as stream:
            stream.write(b"\n" + SYSTEM_CA.read_bytes())

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(leaf, key)
    context.set_alpn_protocols(["http/1.1"])
    return context


def configure_hosts(hosts: list[str]) -> None:
    hosts_file = Path("/etc/hosts")
    lines: list[str] = []
    for line in hosts_file.read_text(encoding="utf-8").splitlines():
        addresses, separator, comment = line.partition("#")
        parts = addresses.split()
        if not set(parts[1:]).intersection(hosts):
            lines.append(line)
        else:
            aliases = [alias for alias in parts[1:] if alias not in hosts]
            if aliases:
                lines.append(" ".join([parts[0], *aliases, separator, comment]).rstrip())
    lines.append(f"{LOOPBACK}\t" + " ".join(hosts))
    hosts_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def serve(
    directory: Path,
    row: dict[str, Any],
    token: str,
    hosts: list[str],
    *,
    stockfish_path: str | None = None,
) -> None:
    os.umask(0o077)
    sys.path.insert(0, str(directory))
    service_mod = importlib.import_module("service")
    request_log = directory / "requests.log"
    request_log.touch()
    out_dir = directory / "state"
    out_dir.mkdir(parents=True, exist_ok=True)
    svc = service_mod.Service(
        row,
        out_dir,
        request_log,
        stockfish_path=stockfish_path,
        signature_path=directory / "engine_signature.json",
    )
    try:
        context = configure_tls(directory, hosts)
        configure_hosts(hosts)
        ThreadingHTTPServer.allow_reuse_address = True
        httpd = ThreadingHTTPServer(
            (LOOPBACK, 443),
            service_mod.make_handler(svc, token),
            bind_and_activate=False,
        )
        httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        httpd.server_bind()
        httpd.server_activate()
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        with (
            httpd,
            (directory / "service.log").open("w", encoding="utf-8") as diagnostics,
            redirect_stderr(diagnostics),
        ):
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                print(f"https://{hosts[0]}", flush=True)
                sys.stdin.read()
                if not thread.is_alive():
                    raise RuntimeError("Chess HTTPS service exited during the episode")
            finally:
                httpd.shutdown()
                thread.join()
    finally:
        game_result = svc.game.result()
        signature = svc.game.engine_signature
        svc.close()

    with request_log.open("rb") as log:
        log.seek(max(0, request_log.stat().st_size - REQUEST_LOG_LIMIT))
        requests_text = log.read().decode("utf-8", "replace")
    print(
        json.dumps(
            {
                "requests": requests_text,
                "endpoint_requests": svc.endpoint_requests,
                "health_requests": svc.health_requests,
                "game_result": game_result,
                "engine_signature": signature,
            }
        ),
        flush=True,
    )
