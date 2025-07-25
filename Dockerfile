FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies (essential for SciPy/PyTorch)
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch FIRST with explicit CPU-only version
RUN pip install \
    --no-cache-dir \
    torch==2.0.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining packages with version pinning
COPY requirements.txt .
RUN pip install \
    --no-cache-dir \
    --disable-pip-version-check \
    -r requirements.txt

# 4. Pre-download model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]