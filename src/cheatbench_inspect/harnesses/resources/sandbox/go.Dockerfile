ARG RUNTIME=codex
FROM ghcr.io/rowan-dauria/task-env-terminus:5f38b4b0ae74 AS terminus
FROM ghcr.io/rowan-dauria/task-env-claude:b771f5ffa542 AS claude
FROM ghcr.io/rowan-dauria/task-env-codex:5f38b4b0ae74 AS codex

FROM ${RUNTIME} AS go
ARG TARGETARCH

# Bubblewrap (bwrap) on PATH for Codex's read-only sandbox preflight.
# No KataGo binary, weights, or sgfmill live in the agent container; they run
# in the separate GPU `engine` service.
RUN --mount=type=bind,source=go-debs-${TARGETARCH}.txt,target=/tmp/packages.txt \
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
