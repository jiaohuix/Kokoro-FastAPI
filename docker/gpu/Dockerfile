FROM --platform=$BUILDPLATFORM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Install Python and other dependencies
# RUN apt-get update -y && \
#     apt-get install -y build-essential python3.10 python3.10-dev  python3-venv espeak-ng espeak-ng-data git libsndfile1 curl ffmpeg g++ cmake && \
#     apt-get clean && rm -rf /var/lib/apt/lists/* && \
#     mkdir -p /usr/share/espeak-ng-data && \
#     ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ && \
#     curl --proto '=https' --tlsv1.2 -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh && \
#     mv /root/.local/bin/uv /usr/local/bin/ && \
#     mv /root/.local/bin/uvx /usr/local/bin/ && \
#     useradd -m -u 1001 appuser && \
#     mkdir -p /app/api/src/models/v1_0 && \
#     chown -R appuser:appuser /app

# Install Python and other dependencies
RUN apt-get update -y && \
    apt-get install -y \
        build-essential \
        python3.10 \
        python3.10-dev \
        python3-venv python3-pip \
        espeak-ng \
        espeak-ng-data \
        git \
        libsndfile1 \
        curl \
        ffmpeg \
        g++ \
        cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/espeak-ng-data && \
    ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ && \
    python3 -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple uv && \
    useradd -m -u 1001 appuser && \
    mkdir -p /app/api/src/models/v1_0 && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies with GPU extras
# RUN uv venv --python 3.10 && \
#     uv sync --extra gpu --no-cache && uv pip install modelscope

RUN uv venv --python 3.10 && \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    uv sync --extra gpu --no-cache && \
    uv pip install modelscope --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh


# Set all environment variables in one go
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    DEVICE="gpu"

ENV DOWNLOAD_MODEL=true
# Download model if enabled
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    python download_model.py --output api/src/models/v1_0; \
    fi

# Run FastAPI server through entrypoint.sh
CMD ["./entrypoint.sh"]
