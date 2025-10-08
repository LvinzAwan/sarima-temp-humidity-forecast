# Gunakan Python slim agar ringan
FROM python:3.11-slim

# Non-interaktif & env yang baik untuk container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependency OS yang sering dibutuhkan numpy/pandas/statsmodels
# (wheel biasanya cukup, tapi ini jaga-jaga)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# Buat direktori kerja
WORKDIR /app

# Copy requirements dulu supaya layer cache efektif
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install -r requirements.txt

# Copy sisa source code
COPY . /app

# Expose port yang digunakan Spaces
EXPOSE 7860

# Command start (Spaces akan eksekusi ini)
CMD ["python", "app.py"]
