# Use official Python image
FROM python:3.11-slim

# Install required build tools for scikit-surprise
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install pip first
RUN python -m ensurepip && pip install --upgrade pip setuptools wheel

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose app port (optional, if you're using Flask or FastAPI)
EXPOSE 5000

# Run your app (change if needed)
CMD ["python", "app.py"]
