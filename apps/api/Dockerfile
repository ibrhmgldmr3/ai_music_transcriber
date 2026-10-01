# Build from the repository root:  docker build -f apps/api/Dockerfile .
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

# CPU-only PyTorch keeps the image small (inference runs ~30x real time on a CPU);
# docker-compose.gpu.yml passes a CUDA index instead.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# torchaudio (Demucs, Beat This!) comes from the same index, so pip keeps this torch.
RUN pip install --no-cache-dir torch torchaudio --index-url ${TORCH_INDEX_URL}

COPY pyproject.toml README.md ./
COPY packages/music-core packages/music-core
COPY ml ml
RUN pip install --no-cache-dir ".[ml]"

COPY apps/api/requirements.txt apps/api/requirements.txt
RUN pip install --no-cache-dir -r apps/api/requirements.txt

COPY apps/api apps/api

# Run unprivileged. The storage and cache directories are created here so the named
# volumes mounted on them inherit this ownership.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/storage /home/app/.cache \
    && chown app:app /data/storage /home/app/.cache
USER app
WORKDIR /repo/apps/api

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
