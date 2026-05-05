# --- Build Stage ---
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Runtime Stage ---
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m wampuser && chown -R wampuser:wampuser /app
USER wampuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/wampuser/.local
ENV PATH=/home/wampuser/.local/bin:$PATH

# Copy source code and pre-configured model
COPY --chown=wampuser:wampuser . .

# Environment Defaults (Research-Proven Safety Thresholds)
ENV PORT=3000
ENV HOST=0.0.0.0
ENV PYTHONPATH=/app/src
ENV ENABLE_ATTENTION_FILTER=true
ENV HF_MODEL_NAME=naranor/SetFit-Multilingual-ONNX-Router-V1
ENV FILTER_MODEL_DIR=/app/model
ENV FILTER_THRESHOLD_MULTIPLIER=1.0

# Adaptive Routing Multipliers (Safe Mode)
ENV FILTER_NEEDLE_MULT=0.99
ENV FILTER_REASONING_MULT=0.95
ENV FILTER_SUMMARY_MULT=0.99

ENV FILTER_MAX_TOKENS=1024

EXPOSE 3000

CMD ["python", "main.py"]
