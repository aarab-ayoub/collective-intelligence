FROM python:3.13.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    pandas \
    numpy \
    scikit-learn \
    psutil \
    paho-mqtt

# Script is expected to be mounted via volumes for development/testing
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "/app/deployment/node_eval.py"]
