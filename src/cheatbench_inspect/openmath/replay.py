"""Stage a private HTTPS replay in a single sample sandbox."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from inspect_ai.util import sandbox

from cheatbench_inspect.openmath.build import RESOURCES
from cheatbench_inspect.sandbox_task import sandbox_service_endpoint

NODE_CA = "/usr/local/share/ca-certificates/site-root.crt"
STARTUP_S = 120


@asynccontextmanager
async def replay_endpoint(
    files: dict[str, bytes],
    hosts: list[str],
    entrypoint: str,
    request_state: dict[str, str] | None,
    **parameters: str,
) -> AsyncGenerator[str, None]:
    environment = sandbox()
    result = await environment.exec(["mktemp", "-d", "/var/tmp/.site.XXXXXXXX"], user="root")
    if not result.success:
        raise RuntimeError(f"Could not create the site directory: {result.stderr}")
    directory = result.stdout.strip()
    for name, content in files.items():
        await environment.write_file(f"{directory}/{name}", content)
    config = {
        "code": (RESOURCES / "start_site.py").read_text() + "\n" + entrypoint,
        "directory": directory,
        "hosts": hosts,
        **parameters,
    }
    async with sandbox_service_endpoint(
        config, startup_timeout_s=STARTUP_S, user="root", grader_state=request_state
    ):
        yield directory
