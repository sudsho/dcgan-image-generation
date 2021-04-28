FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# install OS deps for torchvision / pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ src/
COPY configs/ configs/

ENV DCGAN_CKPT=/app/artifacts/checkpoints/latest.pt
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
