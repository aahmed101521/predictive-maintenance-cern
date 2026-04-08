# ── Dockerfile ───────────────────────────────────────────────────────────────
# Multi-stage build keeps the final image lean
# Stage 1: install dependencies
# Stage 2: copy app code and run

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
# This means rebuilding the image after code changes is fast —
# Docker reuses the cached dependency layer
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY api/ ./api/
COPY models/ ./models/

# Expose the port uvicorn will run on
EXPOSE 8000

# Health check — Docker will mark the container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the FastAPI app with uvicorn
# --host 0.0.0.0 makes it accessible outside the container
# --workers 2 handles concurrent requests
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
