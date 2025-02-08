# Use a CUDA-enabled base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Use a more specific base image version
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    PORT=7860

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser

# Create the cache directory and set permissions
RUN mkdir -p /tmp/huggingface && chown appuser:appuser /tmp/huggingface

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Specify the command to run the application
CMD uvicorn app:asgi_app --host 0.0.0.0 --port $PORT