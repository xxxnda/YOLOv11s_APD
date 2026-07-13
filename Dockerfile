FROM python:3.11-slim

WORKDIR /app

# Install system dependencies yang dibutuhkan
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless || true \
    && pip install --no-cache-dir opencv-python-headless

# Copy semua file aplikasi
COPY . .

# Pastikan folder ada
RUN mkdir -p static/uploads static/results

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --preload
