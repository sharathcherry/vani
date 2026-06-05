FROM python:3.12-slim

# Install system dependencies (e.g., ffmpeg for audio, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models to bake into the image
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; print('Loading Embedder...'); SentenceTransformer('BAAI/bge-small-en-v1.5'); print('Loading Reranker...'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('Models loaded.')"

# Copy application files
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Start Uvicorn server
CMD ["uvicorn", "voice_rag_server_azure:app", "--host", "0.0.0.0", "--port", "8000"]
