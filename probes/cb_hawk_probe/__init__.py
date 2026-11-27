from datetime import datetime, timezone
import gzip
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.solver import solver
from inspect_ai.util import sandbox

ROOT = Path(__file__).parent


@solver
def probe(kind):
    async def solve(state, generate):
        findings = [f"PROBE_SOLVER_START={datetime.now(timezone.utc).isoformat()}"]

        async def run(command, service=None, timeout=60, check=True):
            findings.append(f"PROBE_COMMAND {service or 'default'} {command}")
            result = await sandbox(service).exec(command, timeout=timeout)
            findings.extend((result.stdout, result.stderr, f"PROBE_EXIT={result.returncode}"))
            print("\n".join(findings[-4:]), flush=True)
            if check and not result.success:
                raise RuntimeError("\n".join(findings))
            return result

        print(findings[0], flush=True)
        await run(["date", "-u", "+%FT%T.%NZ"])
        if kind in ("gpu", "gpu-peer"):
            service = None if kind == "gpu" else "engine"
            await run(["nvidia-smi"], service)
            await run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                service,
            )
            await run(["cat", "/usr/local/cuda/version.json"], service, check=False)
            if kind == "gpu-peer":
                no_gpu = await run(
                    [
                        "sh", "-c",
                        'ls -l /dev/nvidia*; command -v nvidia-smi && nvidia-smi; '
                        'test ! -e /dev/nvidiactl && test ! -e /dev/nvidia0',
                    ],
                    check=False,
                )
                await sandbox("engine").write_file(
                    "/tmp/listener", gzip.decompress((ROOT / "listener-amd64.gz").read_bytes())
                )
                await run(
                    ["sh", "-c", "chmod +x /tmp/listener && "
                     "nohup /tmp/listener </dev/null >/tmp/listener.log 2>&1 &"],
                    "engine",
                )
                await run(
                    ["python3", "-c",
                     "import socket, urllib.request; "
                     "print('PROBE_SERVICE_DNS', socket.gethostbyname('engine')); "
                     "reply = urllib.request.urlopen('http://engine:8765/', timeout=15).read(); "
                     "print('PROBE_HTTP_RESPONSE', reply.decode()); "
                     "assert reply == b'engine-ok'"],
                )
                if not no_gpu.success:
                    raise RuntimeError("The default service has NVIDIA devices")
        else:
            parent = {
                "ansible": "585ef6c55e87c10c1ce7d59ebe9c33dd6dbe5afb",
                "protonmail": "738b22f1e8efbb8a0420db86af89fb630a6c9f58",
            }[kind]
            await run(["mkdir", "-p", "/tests"])
            for test_file in (ROOT / "data" / kind).iterdir():
                await sandbox().write_file(f"/tests/{test_file.name}", test_file.read_bytes())
            await sandbox().write_file("/tmp/swe.sh", (ROOT / "swe.sh").read_bytes())
            await run(["bash", "/tmp/swe.sh", parent], timeout=480)
        state.output = ModelOutput.from_content("mockllm/model", "\n".join(findings))
        return state
    return solve


def feasibility_task(kind, layout):
    return Task(
        dataset=[Sample(id=kind, input=f"Run the {kind} feasibility probe.")],
        solver=probe(kind),
        sandbox=("docker", str(ROOT / layout)),
    )


@task
def gpu():
    return feasibility_task("gpu", "gpu-values.yaml")


@task
def gpu_peer():
    return feasibility_task("gpu-peer", "gpu-peer-values.yaml")


@task
def ansible():
    return feasibility_task("ansible", "ansible-compose.yaml")


@task
def protonmail():
    return feasibility_task("protonmail", "protonmail-compose.yaml")
