FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PHYTOTRACE_ENV=production PHYTOTRACE_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir fastapi 'uvicorn[standard]' 'ultralytics==8.4.192' opencv-python-headless \
       pillow pandas scipy scikit-learn pyyaml pymongo cloudinary firebase-admin google-genai httpx python-multipart

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/checkpoints/leaf_segmenter/leaf_unet_best.pt ./models/checkpoints/leaf_segmenter/leaf_unet_best.pt
COPY models/checkpoints/mine_segmenter/best.pt ./models/checkpoints/mine_segmenter/best.pt
ENV PYTHONPATH=/app/src

RUN useradd --create-home phytotrace && mkdir -p /app/logs && chown phytotrace /app/logs
USER phytotrace

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn phytotrace.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
