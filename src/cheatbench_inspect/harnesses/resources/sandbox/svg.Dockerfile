ARG RUNTIME=codex
FROM ghcr.io/rowan-dauria/task-env-terminus:5f38b4b0ae74 AS terminus
FROM ghcr.io/rowan-dauria/task-env-claude:b771f5ffa542 AS claude
FROM ghcr.io/rowan-dauria/task-env-codex:5f38b4b0ae74 AS codex

FROM ${RUNTIME} AS svg
ARG TARGETARCH
# Package paths and SHA-256s from Debian Trixie's signed package indexes.
# Include the transitive packages absent from the pinned base images.
RUN --mount=type=bind,source=svg-debs-${TARGETARCH}.txt,target=/tmp/packages.txt \
    set -eu; \
    work="$(mktemp -d)"; chmod 755 "$work"; cd "$work"; \
    while read -r digest path; do \
        file="${path##*/}"; \
        curl -fsSL --retry 3 "https://deb.debian.org/debian/$path" -o "$file"; \
        echo "$digest  $file" | sha256sum -c -; \
    done < /tmp/packages.txt; \
    cd /; dpkg -i "$work"/*.deb; \
    rm -rf "$work"

RUN --mount=type=bind,source=svg-requirements.txt,target=/tmp/requirements.txt \
    pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && python -c "import cairosvg; cairosvg.svg2png(bytestring=b'<svg width=\"1\" height=\"1\"/>')"
