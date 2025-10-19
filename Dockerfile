# 1️⃣ Base image
FROM python:3.11-slim

# 2️⃣ Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KAGGLE_USERNAME=your_kaggle_username
ENV KAGGLE_KEY=your_kaggle_key

# 3️⃣ Working directory
WORKDIR /app

# 4️⃣ System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 5️⃣ Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copy project files
COPY . .

# 7️⃣ Create artifacts folders
RUN mkdir -p src/artifacts/data \
    src/artifacts/ingested \
    src/artifacts/models \
    src/artifacts/transformed \
    src/artifacts/logs


#  Expose FastAPI port
EXPOSE 8000

#  Run FastAPI with Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
