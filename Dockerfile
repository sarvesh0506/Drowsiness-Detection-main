# Stage 1: Build dependency wheelhouse
FROM python:3.10-slim as builder

WORKDIR /app

# Install system compilation packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies and build wheels
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final lightweight image
FROM python:3.10-slim

WORKDIR /app

# Install runtime OS dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgthread-2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/requirements.txt .

# Ensure local path is mapped
ENV PATH=/root/.local/bin:$PATH
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Copy source code files
COPY . .

# Generate application directory structures
RUN mkdir -p logs models static/uploads static/audio

EXPOSE 5000

# Execute server using Gunicorn multi-threading
# We set workers to 1 and threads to 4, which is optimized for running streaming camera loop concurrency on single-core instances
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
