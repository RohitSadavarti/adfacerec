FROM python:3.9-slim

# 1. Install system dependencies (Still needed for the pre-compiled library to run)
RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy and install standard requirements (including dlib-bin)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. THE TRICK: Install face_recognition without triggering a dlib build
# We already installed dlib-bin, so this will work.
RUN pip install face_recognition --no-deps

# 4. Copy app code
COPY . .

# 5. Start the server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "120"]
