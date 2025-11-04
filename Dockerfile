# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (git, curl optional) and runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir to backend so relative paths in code resolve (e.g., models/)
WORKDIR /app/backend

# Install Python dependencies first for cached layer
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy backend source
COPY backend/ /app/backend/

# Copy model artifacts from repo root into backend/models inside the image
# These are optional; copy only if present to avoid breaking the build
# Note: Docker can't conditionally copy; we create dir and copy known names if they exist
RUN mkdir -p /app/backend/models
COPY Protein_to_Smile.pt /app/backend/models/Protein_to_Smile.pt
COPY Sequence_Generator.pt /app/backend/models/Sequence_Generator.pt
COPY best_mlp_medium_adv.pth /app/backend/models/best_mlp_medium_adv.pth
COPY label_encoder.pkl /app/backend/models/label_encoder.pkl
COPY pca_model.pkl /app/backend/models/pca_model.pkl

# Expose Flask port
EXPOSE 5001

# Default environment for local dev
ENV HOST=0.0.0.0 \
    PORT=5001 \
    DEV_FAKE_OTP=true

# Run server
CMD ["python", "combined_server.py"]
