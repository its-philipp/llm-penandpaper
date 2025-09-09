FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    mpg123 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only files needed to resolve deps first
COPY pyproject.toml README.md ./

# Install project deps using standard pip to avoid uv issues in slim images
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

# Now copy the rest of the app
COPY . .

ENV PORT=8501 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

EXPOSE 8501

CMD ["bash", "run_app.sh"]
