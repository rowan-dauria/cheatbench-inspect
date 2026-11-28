# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/tasks/knowledge_work/Dockerfile
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

ARG RUNTIME=base
FROM ghcr.io/rowan-dauria/task-env-terminus:5f38b4b0ae74 AS terminus
FROM ghcr.io/rowan-dauria/task-env-claude:b771f5ffa542 AS claude
FROM ghcr.io/rowan-dauria/task-env-base:5f38b4b0ae74 AS base

FROM ${RUNTIME} AS office
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer libreoffice-calc libreoffice-impress fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,source=office-requirements.txt,target=/tmp/office-requirements.txt \
    pip install --no-cache-dir --require-hashes -r /tmp/office-requirements.txt
RUN install -d -o user -g user /external /data/cache /data/cache/artifacts \
    && python -c "import fitz, PIL, numpy, docx, pptx, openpyxl, pypdf, reportlab" \
    && soffice --headless --version
