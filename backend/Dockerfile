# dOCRead API (build context = ./backend)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./backend/

ENV HOST=0.0.0.0
ENV PORT=8002
ENV OCR_RUNTIME=auto
ENV OCR_LANG=en,vi
ENV OCR_MAX_SIDE=1280
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8002

CMD ["python", "-m", "backend"]
