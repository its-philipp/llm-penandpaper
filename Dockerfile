FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    mpg123 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv \
    && uv sync

COPY . .

ENV PORT=8501 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

EXPOSE 8501

CMD ["bash", "run_app.sh"]
