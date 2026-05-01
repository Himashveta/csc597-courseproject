# PolyFuzz reproducible build.
#
# This Dockerfile produces an image that:
#   - has the C toolchain needed to build the mock target with gcov + UBSan
#   - has Python 3.12 + coverage.py + pytest
#   - installs PolyFuzz in editable mode
#   - exposes /home/claude/polyfuzz/scripts as the entry surface
#
# Usage:
#   docker build -t polyfuzz .
#   docker run --rm polyfuzz make test          # run unit tests
#   docker run --rm polyfuzz make demo          # 30s smoke fuzz
#   docker run --rm polyfuzz make eval          # 3-trial multi-trial eval
#
# Note: real PyTorch (PYTORCH_TARGET) is NOT installed here. That
# requires a multi-hour custom build, which is outside the scope of
# the artefact docker image. See README.md for the standalone PyTorch
# instructions.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        gcovr \
        python3.12 \
        python3.12-venv \
        python3-pip \
        make \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# We use --break-system-packages because Ubuntu 24 marks the system
# Python as PEP 668 externally-managed; this image is a sandbox so
# system-level installs are appropriate.
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    pip install --break-system-packages --no-cache-dir \
        "coverage>=7.0" \
        "pytest>=7.0"

WORKDIR /opt/polyfuzz
COPY pyproject.toml requirements.txt Makefile ./
COPY src ./src
COPY target ./target
COPY seeds ./seeds
COPY scripts ./scripts
COPY tests ./tests
COPY README.md ./

RUN pip install --break-system-packages --no-cache-dir -e .
RUN make -C target

ENV PYTHONUNBUFFERED=1

# Default: build everything and run unit tests.
CMD ["make", "test"]
