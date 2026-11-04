# Hugging Face Spaces Dockerfile for YouTube Sentiment Analysis API
# Lightweight Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install security updates and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up user according to Hugging Face Spaces standard (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy production requirements first for efficient layer caching
COPY --chown=user:user requirements-prod.txt $HOME/app/requirements-prod.txt

# Install dependencies in user space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy source code and trained ML model artifacts
COPY --chown=user:user src/ $HOME/app/src/
COPY --chown=user:user models/ $HOME/app/models/
COPY --chown=user:user app_api.py $HOME/app/app_api.py

# Create logs directory with appropriate write permissions
RUN mkdir -p $HOME/app/logs

# Expose Hugging Face Space default port
EXPOSE 7860

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch FastAPI with Uvicorn
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
