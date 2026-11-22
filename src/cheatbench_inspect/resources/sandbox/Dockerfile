# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/Dockerfile
# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/terminus_2.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

# The sandbox image every episode runs in, and the single source of truth for
# what is in it. `core/sandbox/build.py` reads THIS FILE and builds from it --
# with `docker build` where a daemon is reachable, and otherwise by replaying
# the same steps through enroot and exporting a squashfs. There is no second
# description of the image anywhere.
#
# What this is: the TASK ENVIRONMENT, and nothing else. An interpreter and the
# libraries a task's work needs. From inside it is an ordinary Python container.
#
# What this is NOT:
#
#   no harness source, no prompts, no judge rubric, no answer keys, no dataset.
#   The runner, the staging module, the task's grader and the row are streamed
#   in on stdin at exec time (core/trial.py) and never touch the container
#   filesystem, so an agent that runs `ls /`, `find / -name '*.md'` or reads
#   /proc finds nothing that identifies this eval.
#
#   no agent, and nothing the agent is made of. Baking
#   `npm install -g @anthropic-ai/claude-code` in welded one vendor to the
#   sandbox and would mean a second image the day we run Codex. The same goes
#   for the vendor's Python SDK -- an agent running `pip list` should not find
#   the harness's own `claude-agent-sdk` sitting in its environment -- and for
#   Node, which is the CLI's dependency and not the environment's. All three
#   are installed at episode start by the adapter (core/agents/installed.py:
#   install_check / install_script), which is why only `curl` is here.
#
# Anything an adapter installs may also be pre-baked here as a pure CACHE: the
# adapter's check succeeds, its install never runs, and the episode starts
# exactly as before. So this split is a performance choice, never a correctness
# one. Measured cold start for all three: 33 s per episode, against a 900-3600 s
# episode budget (Node 22 via nvm 12.6 s, the SDK pins ~15 s, the CLI 4.9 s --
# and Node 22 makes the CLI install faster than the Node 20 that used to be
# here, 4.9 s against 10.0 s, so moving Node out costs ~7 s net).
#
# Anonymity is a regression test, not a convention: tests/test_image_anonymity.py.

# Pinned by digest so a re-pushed tag cannot change the build.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS task

# curl/git: the network and VCS tools an ordinary Python box has. curl is also
# what the agent adapter fetches nvm with -- a general-purpose tool, unlike a
# Node runtime, which would say something about what runs here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Task-side libraries only: what a row's work plausibly needs. The original
# installs numpy, sympy, mpmath, matplotlib, pillow==12.3.0, requests==2.34.2 and
# pyyaml==6.0.3 by name; task-requirements.txt locks that set with hashes. It is
# bind-mounted for this step only, so no file of ours -- and none of its comments
# about this eval -- lands in the image. Regenerate it from task-requirements.in:
#   uv pip compile task-requirements.in --python-version 3.11 --python-platform linux --generate-hashes -o task-requirements.txt
RUN --mount=type=bind,source=task-requirements.txt,target=/tmp/task-requirements.txt \
    pip install --no-cache-dir --require-hashes -r /tmp/task-requirements.txt

# The agent is not root. It runs as a stock container user, uid 1000 named
# "user", with an empty home, as the original's sandbox runtime names it.
# https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/sandbox/runtime.py#L104-L139
RUN useradd --uid 1000 --user-group --no-create-home --home-dir /home/user --shell /bin/bash user \
    && install -d -m 755 -o user -g user /home/user

# A neutral, non-identifying working directory for the agent.
RUN install -d -o user -g user /workspace
WORKDIR /workspace

# No ENTRYPOINT and no CMD: the runner is streamed in on stdin at exec time.
# No ENV either -- PYTHONUNBUFFERED is passed per episode by core/sandbox, and
# ANTHROPIC_API_KEY is passed by name so it never appears in a command line.

# Cache the original harness runtime only for Terminus-2 episodes. Harbor, which
# runs as the agent's user, keeps its agent logs under /logs, and its Python must
# live outside root's home.
FROM task AS terminus
ARG HARBOR_COMMIT=4641fd590f190411c57e1cbd824a9f6786cbe92e
ARG HARBOR_SHA256=ca4079f8189f290134c8619c66a131b21a598893e4d454cceb64a8f53b8cfc79
RUN set -eu; \
    apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*; \
    mkdir -p /opt/terminus-runtime /tmp/harbor; \
    python -m venv /opt/terminus-runtime/installer; \
    /opt/terminus-runtime/installer/bin/pip install --no-cache-dir uv==0.9.9 asciinema==2.4.0; \
    ln -s /opt/terminus-runtime/installer/bin/asciinema /usr/local/bin/asciinema; \
    cd /tmp/harbor; \
    curl -fsSL --retry 3 "https://codeload.github.com/harbor-framework/harbor/tar.gz/${HARBOR_COMMIT}" -o source.tar.gz; \
    echo "${HARBOR_SHA256}  source.tar.gz" | sha256sum -c -; \
    tar -xzf source.tar.gz --strip-components=1; \
    UV_PYTHON_INSTALL_DIR=/opt/terminus-runtime/python UV_PROJECT_ENVIRONMENT=/opt/terminus-runtime/.venv /opt/terminus-runtime/installer/bin/uv sync --frozen --no-dev --no-editable --package harbor --python 3.12.11; \
    /opt/terminus-runtime/installer/bin/uv pip install --python /opt/terminus-runtime/.venv/bin/python --no-deps pillow==11.3.0; \
    cp uv.lock /opt/terminus-runtime/uv.lock; \
    echo "${HARBOR_COMMIT}" > /opt/terminus-runtime/commit; \
    cd /; rm -rf /tmp/harbor /root/.cache/uv; \
    install -d -o user -g user /logs

FROM task AS default
