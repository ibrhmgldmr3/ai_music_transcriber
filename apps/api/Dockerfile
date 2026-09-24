# Build from the repository root:  docker build -f apps/api/Dockerfile .
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

# CPU-only PyTorch keeps the image small; use the CUDA index URL for GPU inference.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY packages/music-core packages/music-core
COPY ml ml
RUN pip install --no-cache-dir ".[ml]"

COPY apps/api/requirements.txt apps/api/requirements.txt
RUN pip install --no-cache-dir -r apps/api/requirements.txt

COPY apps/api apps/api

# Run unprivileged. The storage directory is created here so the named volume mounted
# on it inherits this ownership.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/storage \
    && chown app:app /data/storage
USER app
WORKDIR /repo/apps/api

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
