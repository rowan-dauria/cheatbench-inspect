"""Root-owned HTTPS frontend on `default` forwarding /harness/call to the `engine` service."""

from __future__ import annotations

import hmac
import json
import os
import socket
import ssl
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

LOOPBACK = "127.0.0.1"
SYSTEM_CA = Path("/usr/local/share/ca-certificates/site-root.crt")
MAX_REQUEST_BYTES = 1048576


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


def _make_handler(
    upstream_url: str,
    grader_token: str,
    engine_token: str,
    stats: dict[str, Any],
    lock: threading.Lock,
) -> type[BaseHTTPRequestHandler]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    target_url = upstream_url.rstrip("/") + "/harness/call"

    class Handler(BaseHTTPRequestHandler):
        server_version = "agents/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _reject(self) -> None:
            with lock:
                stats["rejected_requests"] += 1
            body = json.dumps({"error": "Unknown episode service endpoint."}).encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/harness/call" or not hmac.compare_digest(
                self.headers.get("X-Harness-Token", ""), grader_token
            ):
                self._reject()
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size")
                payload = self.rfile.read(length)
            except ValueError:
                data = json.dumps(
                    {
                        "content": [{"type": "text", "text": "Invalid Go tool request."}],
                        "isError": True,
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            req = urllib.request.Request(
                target_url,
                data=payload,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "X-Harness-Token": engine_token,
                },
            )
            try:
                with opener.open(req, timeout=900) as resp:
                    status = resp.status
                    data = resp.read()
            except urllib.error.HTTPError as exc:
                status = exc.code
                data = exc.read()
            except Exception as exc:
                status = 200
                data = json.dumps(
                    {
                        "infrastructure_error": f"Go engine service unreachable: {type(exc).__name__}: {exc}"
                    }
                ).encode("utf-8")
            with lock:
                stats["forwarded_requests"] += 1
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._reject()

        def do_PUT(self) -> None:
            self._reject()

        def do_DELETE(self) -> None:
            self._reject()

        def do_PATCH(self) -> None:
            self._reject()

        def do_HEAD(self) -> None:
            self._reject()

        def do_OPTIONS(self) -> None:
            self._reject()

    return Handler


def serve(
    directory: Path,
    hosts: list[str],
    upstream_url: str,
    grader_token: str,
    engine_token: str,
) -> None:
    os.umask(0o077)
    context = configure_tls(directory, hosts)
    configure_hosts(hosts)
    stats: dict[str, Any] = {"forwarded_requests": 0, "rejected_requests": 0}
    lock = threading.Lock()
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(
        (LOOPBACK, 443),
        _make_handler(upstream_url, grader_token, engine_token, stats, lock),
        bind_and_activate=False,
    )
    httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    httpd.server_bind()
    httpd.server_activate()
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    with (
        httpd,
        (directory / "proxy.log").open("w", encoding="utf-8") as diagnostics,
        redirect_stderr(diagnostics),
    ):
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            print(f"https://{hosts[0]}", flush=True)
            sys.stdin.read()
            if not thread.is_alive():
                raise RuntimeError("Go HTTPS proxy exited during the episode")
        finally:
            httpd.shutdown()
            thread.join()
    print(json.dumps(stats), flush=True)
