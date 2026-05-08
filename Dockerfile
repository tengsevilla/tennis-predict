FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# libgomp1 is needed for XGBoost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway dynamically assigns a PORT environment variable
ENV PORT=8000

CMD uvicorn api:app --host 0.0.0.0 --port $PORT