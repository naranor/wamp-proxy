# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# Set Python path to find src
ENV PYTHONPATH=/app

ENV PORT=3000
ENV UPSTREAM_URL=https://api.openai.com
ENV ENABLE_ATTENTION_FILTER=true
ENV FILTER_THRESHOLD_MULTIPLIER=1.5
ENV FILTER_MAX_TOKENS=1024

EXPOSE 3000

CMD ["python", "main.py"]
