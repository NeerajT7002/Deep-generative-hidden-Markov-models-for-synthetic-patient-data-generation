FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and trained outputs
COPY . .

# Expose port (default Flask port)
EXPOSE 5000

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Run with Gunicorn production WSGI server
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:app --workers 2 --timeout 120"]
