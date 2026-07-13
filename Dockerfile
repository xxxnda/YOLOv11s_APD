FROM python:3.11-slim

WORKDIR /app

# Install system dependencies yang dibutuhkan
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Langkah 1: Install semua requirements (ultralytics akan pull opencv-python)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Langkah 2: Hapus SEMUA varian opencv yang mungkin ter-install
#            (termasuk opencv-python yang ditarik oleh ultralytics)
RUN pip uninstall -y \
        opencv-python \
        opencv-contrib-python \
        opencv-python-headless \
        opencv-contrib-python-headless \
    2>/dev/null || true

# Langkah 3: Install ulang HANYA versi headless (aman untuk server tanpa display)
RUN pip install --no-cache-dir "opencv-python-headless==4.10.0.84"

# Copy semua file aplikasi
COPY . .

# Pastikan folder ada
RUN mkdir -p static/uploads static/results

EXPOSE 8080

CMD ["/bin/sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 300 --preload"]
