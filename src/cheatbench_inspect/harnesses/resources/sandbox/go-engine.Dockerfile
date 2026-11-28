# Isolated GPU Go engine container (linux/amd64, CUDA 12.1 + cuDNN 8).
# Runs KataGo v1.18.2 (MIT) + pinned b18c384nbt network (KataGo Neural Network License)
# + sgfmill==1.1.1 (MIT) + libseccomp2 (LGPL-2.1) for Landlock+seccomp policy workers.
# Digest matches the verified Hawk GPU probe (nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:810756cab1c28ce693499a5c2ebb66f6d10a61d026998c8606bad449643a4c49 AS go-engine

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libseccomp2 \
        python3 \
        python3-venv \
        unzip; \
    rm -rf /var/lib/apt/lists/*

# Isolated venv with hash-locked sgfmill==1.1.1 plus pinned KataGo v1.18.2 and network.
RUN --mount=type=bind,source=go-engine-requirements.txt,target=/tmp/go-engine-requirements.txt \
    set -eu; \
    install -d -o root -g root -m 700 /opt/engine /opt/engine/katago; \
    python3 -m venv /opt/engine/venv; \
    /opt/engine/venv/bin/pip install --no-cache-dir --require-hashes -r /tmp/go-engine-requirements.txt; \
    work="$(mktemp -d)"; \
    curl -fsSL --retry 3 \
        "https://github.com/lightvector/KataGo/releases/download/v1.18.2/katago-v1.18.2-cuda12.1-cudnn8.9.7-linux-x64.zip" \
        -o "$work/katago.zip"; \
    echo "16c69f42291fe8c6d196d722d92299a5b04de852611d6c022a0dd9e0e83b5688  $work/katago.zip" | sha256sum -c -; \
    unzip -q "$work/katago.zip" -d "$work/unzipped"; \
    chmod 0755 "$work/unzipped/katago"; \
    (cd "$work/unzipped" && ./katago --appimage-extract >/dev/null); \
    mv "$work/unzipped/squashfs-root" /opt/engine/katago/squashfs-root; \
    curl -fsSL --retry 3 \
        "https://raw.githubusercontent.com/lightvector/KataGo/v1.18.2/LICENSE" \
        -o /opt/engine/LICENSE-katago; \
    echo "ead22455d923920cd152546d89e4f415c94d6b580aa109fda3b57a69c57c32e0  /opt/engine/LICENSE-katago" | sha256sum -c -; \
    curl -fsSL --retry 3 \
        "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz" \
        -o /opt/engine/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz; \
    echo "9d7a6afed8ff5b74894727e156f04f0cd36060a24824892008fbb6e0cba51f1d  /opt/engine/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz" | sha256sum -c -; \
    rm -rf "$work"; \
    chown -R root:root /opt/engine; \
    chmod 700 /opt/engine; \
    /opt/engine/katago/squashfs-root/AppRun version | grep -q "KataGo v1.18.2"
