FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for efficient caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and model
COPY . .

# Expose default port (Hugging Face Spaces standard port is 7860)
EXPOSE 7860

# Command runs uvicorn using PORT env var if provided (Render), default 7860 (HF Spaces)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
