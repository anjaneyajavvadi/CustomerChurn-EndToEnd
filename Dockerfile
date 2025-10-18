# Use lightweight Python image
FROM python:3.10-slim

# Copy environment variables
ENV KAGGLE_USERNAME=${KAGGLE_USERNAME}
ENV KAGGLE_KEY=${KAGGLE_KEY}


# Set working directory inside container
WORKDIR /app

# Copy requirements file first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Run the FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
