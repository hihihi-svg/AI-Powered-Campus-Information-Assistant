FROM python:3.11-slim

# Create a non-root user with UID 1000 required for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface/hub \
    TORCH_HOME=/home/user/.cache/torch

# Install system dependencies needed for PyMuPDF, RapidOCR / OpenCV, and curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

# Pre-cache the sentence transformer embedding model during build for fast offline startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || true

# Copy application source code, assets, and processed corpus
COPY --chown=user:user src ./src
COPY --chown=user:user static ./static
COPY --chown=user:user config.yaml ./config.yaml
COPY --chown=user:user data/processed ./data/processed
COPY --chown=user:user data/queries ./data/queries
COPY --chown=user:user reports ./reports

# Prepare writable cache directories and ensure proper permissions
RUN mkdir -p /app/data/cache && chown -R user:user /app /home/user

# Switch to non-root user
USER user

ENV CAMPUS_DATASET_PATH=/app/data/processed/campus_documents.jsonl \
    CAMPUS_EMBEDDINGS_PATH=/app/data/processed/corpus_semantic_embeddings.npy

EXPOSE 7860

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
