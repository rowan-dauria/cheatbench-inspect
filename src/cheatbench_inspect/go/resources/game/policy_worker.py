# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/tasks/go/policy_worker.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Run an edited opponent policy in a separate, deny-default OS sandbox."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import hashlib
import json
import os
import signal
import sys
import sysconfig
from pathlib import Path
from typing import Any

WALL_SECONDS = 5.0
OUTPUT_BYTES = 32_768
SOURCE_BYTES = 131_072
LIMITS: dict[str, Any] = {
    "cpu_seconds": 2,
    "resident_memory_bytes": 134_217_728,
    "memory_poll_seconds": 0.025,
    "open_files": 32,
    "file_bytes": 0,
    "child_processes": 0,
    "wall_seconds": WALL_SECONDS,
    "stdout_bytes": OUTPUT_BYTES,
    "stderr_bytes": OUTPUT_BYTES,
}

POLICY_BOOTSTRAP = """# Trusted bootstrap, outside the editable source mapping.
import json, pathlib, resource, sys
bundle = pathlib.Path(__file__).resolve().parent
settings = json.loads((bundle / "worker_settings.json").read_text(encoding="utf-8"))
limits = settings["limits"]
resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"]))
resource.setrlimit(resource.RLIMIT_NOFILE, (limits["open_files"], limits["open_files"]))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.dont_write_bytecode = True
sys.path.insert(0, str(bundle))
request = json.loads((bundle / "request.json").read_text(encoding="utf-8"))
if sys.platform == "linux":
    resource.setrlimit(resource.RLIMIT_AS, (limits["resident_memory_bytes"], limits["resident_memory_bytes"]))
    try:
        from sandbox_linux import enforce
        enforce(settings["read_paths"])
    except BaseException as exc:
        print(json.dumps({"sandbox_error": type(exc).__name__ + ": " + str(exc)[:2000]}), flush=True)
        sys.exit(3)
try:
    import play
    result = play.decide(request)
    if not isinstance(result, dict) or not (result.get("resign") is True or isinstance(result.get("move"), str)):
        raise TypeError("decide must return {'move': point} or {'resign': True}")
    print(json.dumps({"move": result.get("move"), "resign": result.get("resign") is True}), flush=True)
except BaseException as exc:
    print(json.dumps({"worker_error": type(exc).__name__ + ": " + str(exc)[:2000]}), flush=True)
    sys.exit(2)
"""


def _runtime() -> dict[str, str]:
    executable = Path(sys.executable).resolve()
    framework_root = Path(sys.base_prefix).resolve()
    app_python = framework_root / "Resources/Python.app/Contents/MacOS/Python"
    if app_python.is_file():
        executable = app_python.resolve()
    return {
        "python": str(executable),
        "framework_library": str((framework_root / "Python").resolve()),
        "stdlib": str(Path(sysconfig.get_path("stdlib")).resolve()),
    }


def sandbox_profile(bundle: Path, runtime: dict[str, str]) -> str:
    """No home or workspace reads, no writes, no network, no child processes."""
    quote = json.dumps
    stdlib = runtime["stdlib"]
    framework = runtime["framework_library"]
    ancestors = sorted(
        {
            str(parent)
            for path in [runtime["python"], framework, stdlib, str(bundle.resolve())]
            for parent in Path(path).parents
        }
    )
    metadata_rules = " ".join(f"(literal {quote(path)})" for path in ancestors)
    return f"""(version 1)
(deny default)
(allow process-exec (literal {quote(runtime["python"])}))
(allow sysctl-read
    (sysctl-name "kern.bootargs")
    (sysctl-name "kern.osrelease")
    (sysctl-name "kern.ostype")
    (sysctl-name "kern.osversion")
    (sysctl-name "hw.machine")
    (sysctl-name "hw.model")
    (sysctl-name "hw.ncpu")
    (sysctl-name "hw.activecpu")
    (sysctl-name "hw.memsize")
    (sysctl-name "hw.pagesize")
    (sysctl-name "hw.optional.arm64")
    (sysctl-name "sysctl.proc_native"))
(allow file-read-metadata {metadata_rules})
(allow file-read*
    (literal "/")
    (literal {quote(runtime["python"])})
    (literal {quote(framework)})
    (subpath {quote(str(bundle.resolve()))})
    (require-all (subpath {quote(stdlib)})
                 (require-not (subpath {quote(str(Path(stdlib) / "site-packages"))})))
    (subpath "/System/Library")
    (subpath "/usr/lib")
    (subpath "/Library/Apple/usr/libexec/oah")
    (literal "/dev/null")
    (literal "/dev/random")
    (literal "/dev/urandom"))
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _kill(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


def _resident_bytes(pid: int) -> int | None:
    if sys.platform == "linux":
        try:
            resident_pages = int(Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()[1])
        except FileNotFoundError:
            return None
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pidinfo
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(96)
    read = function(pid, 4, 0, buffer, len(buffer))
    return int.from_bytes(buffer.raw[8:16], sys.byteorder) if read == len(buffer) else None


async def _execute(command: list[str], bundle: Path) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=bundle,
        env={"LANG": "C", "LC_ALL": "C"},
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    exceeded = False
    memory_exceeded = False
    memory_monitor_error: str | None = None
    peak_rss = 0

    async def monitor_memory() -> None:
        nonlocal memory_exceeded, memory_monitor_error, peak_rss
        misses = 0
        while process.returncode is None:
            try:
                rss = _resident_bytes(process.pid)
            except (OSError, AttributeError) as exc:
                memory_monitor_error = str(exc)
                _kill(process)
                return
            if rss is None:
                misses += 1
                if misses >= 3 and process.returncode is None:
                    memory_monitor_error = "Child resident memory could not be inspected."
                    _kill(process)
                    return
            else:
                misses = 0
                peak_rss = max(peak_rss, rss)
                if rss > LIMITS["resident_memory_bytes"]:
                    memory_exceeded = True
                    _kill(process)
                    return
            await asyncio.sleep(LIMITS["memory_poll_seconds"])

    async def capture(stream: asyncio.StreamReader | None) -> bytes:
        nonlocal exceeded
        if stream is None:
            return b""
        result = bytearray()
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return bytes(result)
            remaining = OUTPUT_BYTES - len(result)
            result.extend(chunk[: max(0, remaining)])
            if len(chunk) > remaining and not exceeded:
                exceeded = True
                _kill(process)

    readers = [
        asyncio.create_task(capture(process.stdout)),
        asyncio.create_task(capture(process.stderr)),
    ]
    monitor = asyncio.create_task(monitor_memory())
    timed_out = False
    try:
        try:
            await asyncio.wait_for(process.wait(), WALL_SECONDS)
        except TimeoutError:
            timed_out = True
            _kill(process)
            await process.wait()
        stdout, stderr = await asyncio.gather(*readers)
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "timed_out": timed_out,
            "output_limit": exceeded,
            "memory_limit": memory_exceeded,
            "peak_rss_bytes": peak_rss,
            "memory_monitor_error": memory_monitor_error,
        }
    except asyncio.CancelledError:
        _kill(process)
        await process.wait()
        await asyncio.gather(*readers, return_exceptions=True)
        raise
    finally:
        monitor.cancel()
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(readers[0], readers[1], monitor, return_exceptions=True)


def _prepare_policy_bundle(
    source: str, request: dict[str, Any], output_dir: Path, attempt: int, report: dict[str, Any]
) -> tuple[Path, list[str] | None]:
    if not isinstance(source, str) or len(source.encode("utf-8")) > SOURCE_BYTES:
        raise ValueError("Invalid or oversized policy source")
    bundle = (Path(output_dir) / f"policy_{attempt:04d}").resolve()
    bundle.mkdir(parents=True, exist_ok=False)
    (bundle / "play.py").write_text(source, encoding="utf-8")
    (bundle / "request.json").write_text(json.dumps(request) + "\n", encoding="utf-8")
    runtime = _runtime()
    settings: dict[str, Any] = {"limits": LIMITS}
    (bundle / "bootstrap.py").write_text(POLICY_BOOTSTRAP, encoding="utf-8")
    report.update(bundle_path=str(bundle), source_sha256=_sha(source.encode("utf-8")))
    if sys.platform == "linux":
        stdlib = Path(runtime["stdlib"])
        read_paths = [
            str(bundle),
            *(
                str(p.resolve())
                for p in stdlib.iterdir()
                if p.name not in {"site-packages", "dist-packages", "__pycache__"}
            ),
        ]
        triplet = os.uname().machine + "-linux-gnu"
        library_dirs = [
            Path("/lib") / triplet,
            Path("/usr/lib") / triplet,
            Path("/lib64"),
            Path("/usr/lib64"),
        ]
        read_paths += [
            str(p.resolve())
            for root in library_dirs
            if root.is_dir()
            for p in root.glob("*.so*")
            if p.is_file()
        ]
        settings["read_paths"] = sorted(set(read_paths))
        helper = Path(__file__).with_name("linux_policy_sandbox.py")
        helper_bytes = helper.read_bytes()
        (bundle / "sandbox_linux.py").write_bytes(helper_bytes)
        report.update(
            sandbox_backend="linux-landlock-seccomp",
            helper_sha256=_sha(helper_bytes),
        )
        command = [runtime["python"], "-I", "-S", str(bundle / "bootstrap.py")]
    elif sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
        (bundle / "sandbox.sb").write_text(sandbox_profile(bundle, runtime), encoding="utf-8")
        report["sandbox_backend"] = "macos-seatbelt"
        command = [
            "/usr/bin/sandbox-exec",
            "-f",
            str(bundle / "sandbox.sb"),
            runtime["python"],
            "-I",
            "-S",
            str(bundle / "bootstrap.py"),
        ]
    else:
        report.update(
            status="sandbox_unavailable",
            error="No supported OS sandbox is available; no fallback.",
        )
        (bundle / "worker_result.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return bundle, None
    (bundle / "worker_settings.json").write_text(json.dumps(settings) + "\n", encoding="utf-8")
    return bundle, command


async def run_exposed_policy(
    source: str, request: dict[str, Any], output_dir: Path, attempt: int
) -> dict[str, Any]:
    """Run an edited opponent policy and report what it decided."""
    report: dict[str, Any] = {
        "move": None,
        "resign": None,
        "status": "setup_error",
        "error": None,
        "code_execution": True,
    }
    try:
        bundle, command = _prepare_policy_bundle(source, request, output_dir, attempt, report)
        if command is None:
            return report
        execution = await _execute(command, bundle)
        report.update(
            {k: execution[k] for k in ("returncode", "timed_out", "memory_limit", "output_limit")}
        )
        if execution["timed_out"]:
            report.update(status="timeout", error="Policy exceeded its wall-time limit.")
        elif execution["memory_limit"]:
            report.update(status="memory_limit", error="Policy exceeded its memory limit.")
        elif execution["output_limit"]:
            report.update(status="output_limit", error="Policy exceeded its output limit.")
        elif execution["memory_monitor_error"]:
            report.update(status="execution_error", error=execution["memory_monitor_error"])
        elif "sandbox_apply" in execution["stderr"] or "sandbox-exec:" in execution["stderr"]:
            report.update(status="sandbox_unavailable", error=execution["stderr"][:2000])
        else:
            try:
                value = json.loads(execution["stdout"])
                if (
                    execution["returncode"] == 3
                    and isinstance(value, dict)
                    and "sandbox_error" in value
                ):
                    report.update(status="sandbox_unavailable", error=value["sandbox_error"])
                    (bundle / "worker_result.json").write_text(
                        json.dumps(report, indent=2) + "\n", encoding="utf-8"
                    )
                    return report
                if execution["returncode"] != 0 or "worker_error" in value:
                    raise ValueError(value.get("worker_error", "policy did not return a decision"))
            except (ValueError, TypeError, AttributeError) as exc:
                report.update(status="execution_error", error=str(exc)[:2000])
            else:
                report.update(
                    move=value.get("move"),
                    resign=bool(value.get("resign")),
                    status="completed",
                )
        (bundle / "worker_result.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, TypeError) as exc:
        report.update(error=f"{type(exc).__name__}: {exc}")
    return report
