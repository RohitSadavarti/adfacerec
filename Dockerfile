# Use Python 3.14 (slim version is faster)
FROM python:3.14.3-slim

# 1. Install the missing system libraries (The Fix!)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up the app
WORKDIR /app

# 3. Copy requirements and install them
COPY requirements.txt .
# We install cmake again via pip just to be safe
RUN pip install cmake
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the code
COPY . .

# 5. Start the server (with a longer timeout for face processing)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "120"]
