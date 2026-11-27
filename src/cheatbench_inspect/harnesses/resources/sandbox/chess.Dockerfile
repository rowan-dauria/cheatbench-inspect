ARG RUNTIME=codex
FROM ghcr.io/rowan-dauria/task-env-terminus:5f38b4b0ae74 AS terminus
FROM ghcr.io/rowan-dauria/task-env-claude:b771f5ffa542 AS claude
FROM ghcr.io/rowan-dauria/task-env-codex:5f38b4b0ae74 AS codex

FROM ${RUNTIME} AS chess
ARG TARGETARCH

# Bubblewrap (bwrap) on PATH for Codex's read-only sandbox preflight.
# Package path and SHA-256 from Debian Trixie's signed package indexes.
RUN --mount=type=bind,source=chess-debs-${TARGETARCH}.txt,target=/tmp/packages.txt \
    set -eu; \
    work="$(mktemp -d)"; chmod 755 "$work"; cd "$work"; \
    while read -r digest path; do \
        file="${path##*/}"; \
        curl -fsSL --retry 3 "https://deb.debian.org/debian/$path" -o "$file"; \
        echo "$digest  $file" | sha256sum -c -; \
    done < /tmp/packages.txt; \
    cd /; dpkg -i "$work"/*.deb; \
    rm -rf "$work"; \
    bwrap --version

# Root-only game service runtime: isolated python-chess venv and pinned Stockfish 19
# (GPL-3.0, https://github.com/official-stockfish/Stockfish/releases/tag/sf_19).
# Mode 0700 owned by root keeps the engine and chess package unreachable from uid 1000.
RUN --mount=type=bind,source=chess-requirements.txt,target=/tmp/chess-requirements.txt \
    set -eu; \
    case "$TARGETARCH" in \
      amd64) name="stockfish-linux-x86-64-universal"; digest="9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f" ;; \
      arm64) name="stockfish-linux-arm64-universal"; digest="fe26cfd1d9db4c8af3d21e24d9ff34cacb31c1f940085a7583da11796f2bac01" ;; \
      *) echo "Unsupported TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    install -d -o root -g root -m 700 /opt/.svc /opt/.svc/bin; \
    python3 -m venv /opt/.svc/venv; \
    /opt/.svc/venv/bin/pip install --no-cache-dir --require-hashes -r /tmp/chess-requirements.txt; \
    work="$(mktemp -d)"; \
    curl -fsSL --retry 3 "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/${name}.tar.gz" -o "$work/sf.tar.gz"; \
    echo "$digest  $work/sf.tar.gz" | sha256sum -c -; \
    tar -xzf "$work/sf.tar.gz" -C "$work" "stockfish/${name}" "stockfish/Copying.txt"; \
    install -o root -g root -m 755 "$work/stockfish/${name}" /opt/.svc/bin/engine; \
    install -o root -g root -m 644 "$work/stockfish/Copying.txt" /opt/.svc/Copying.txt; \
    rm -rf "$work"; \
    chown -R root:root /opt/.svc; \
    chmod 700 /opt/.svc; \
    printf 'uci\nquit\n' | /opt/.svc/bin/engine | grep -qx 'id name Stockfish 19'
