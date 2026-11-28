"""The agent's identity in an episode sandbox and the launcher for in-sandbox code.

The original drops the agent to a stock container user, uid 1000 named "user"
with home /home/user, and runs the harness and its task services as that user.
It owns the staged files and its workspace.
https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/sandbox/runtime.py#L104-L139
"""

from pathlib import Path, PurePosixPath

from inspect_ai.util import sandbox

SANDBOX_RESOURCES = Path(__file__).parent / "resources" / "sandbox"

AGENT_USER = "user"
AGENT_HOME = f"/home/{AGENT_USER}"
AGENT_ENV = {"HOME": AGENT_HOME, "USER": AGENT_USER, "LOGNAME": AGENT_USER}
ROOT_USER = "root"
ROOT_ENV = {"HOME": "/root", "USER": ROOT_USER, "LOGNAME": ROOT_USER}


def user_env(user: str) -> dict[str, str]:
    """Return the default HOME/USER/LOGNAME environment for the sandbox user."""
    if user == AGENT_USER:
        return AGENT_ENV
    if user == ROOT_USER:
        return ROOT_ENV
    raise ValueError(f"Unsupported sandbox user: {user!r}")


# The agent can read every process's command line, so nothing on it may
# identify the eval. The program and its inputs arrive as JSON on stdin, as in
# the original's runner bootstrap; the program reads its inputs from PAYLOAD.
# https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/sandbox/runtime.py#L158-L163
BOOTSTRAP = (
    'import sys,json;d=json.load(sys.stdin);exec(compile(d.pop("code"),"<r>","exec"),{"PAYLOAD":d})'
)


# Keep stdin open for episode messages after the initial, newline-delimited payload.
STREAM_BOOTSTRAP = 'import sys,json;d=json.loads(sys.stdin.readline());exec(compile(d.pop("code"),"<r>","exec"),{"PAYLOAD":d})'


def agent_owned_roots(staged: list[str]) -> list[str]:
    """The paths to hand to the agent's user after Inspect stages files as root.

    That is /workspace and each home or /tmp subdirectory a staged file lies
    under. A staged file anywhere else is handed over alone, so no system
    directory changes owner.
    """
    roots = {"/workspace"}
    for path in staged:
        parts = PurePosixPath(path).parts
        if parts[1] == "workspace":
            continue
        if parts[1] in {"home", "tmp"} and len(parts) > 3:
            roots.add(str(PurePosixPath(*parts[:3])))
        else:
            roots.add(path)
    return sorted(roots)


async def hand_staged_files_to_agent(staged: list[str]) -> None:
    """Give the agent's user ownership of its workspace and staged files."""
    owner = f"{AGENT_USER}:{AGENT_USER}"
    result = await sandbox().exec(["chown", "-R", owner, *agent_owned_roots(staged)])
    if not result.success:
        raise RuntimeError(f"Could not hand staged files to the agent: {result.stderr}")
