# CPU-only Go engine test double for local Docker tests (no NVIDIA GPU required).
# Runs the exact same GoGame, Episode, EpisodeEngineAPI, TournamentSurfaces,
# engine_signature verification, and Landlock+seccomp policy_worker stack with
# a deterministic KataGo analysis protocol test double.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS go-engine-double

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends libseccomp2; \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=go-engine-requirements.txt,target=/tmp/go-engine-requirements.txt \
    set -eu; \
    install -d -o root -g root -m 700 /opt/engine; \
    python3 -m venv /opt/engine/venv; \
    /opt/engine/venv/bin/pip install --no-cache-dir --require-hashes -r /tmp/go-engine-requirements.txt

COPY katago_double.py /opt/engine/katago-double
RUN chmod 0755 /opt/engine/katago-double && \
    printf 'test-double-network\n' > /opt/engine/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz && \
    chmod 0600 /opt/engine/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz && \
    /opt/engine/katago-double version | grep -q "KataGo v1.18.2"

ENV GO_ENGINE_TEST_DOUBLE=1
