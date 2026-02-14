FROM python:3.9-slim

# --- THE MAGIC FIX ---
# Force cmake to use only 1 CPU core for compilation.
# This prevents the "Out of Memory" crash.
ENV CMAKE_BUILD_PARALLEL_LEVEL=1

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements
COPY requirements.txt .

# 4. Install dependencies
# (This step will now take about 5-10 minutes, but it won't crash)
RUN pip install cmake
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the code
COPY . .

# 6. Start the server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "120"]
