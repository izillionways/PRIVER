FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md LICENSE /tmp/priver/
COPY src /tmp/priver/src
RUN python3 -m pip install --no-deps /tmp/priver

CMD ["bash"]
