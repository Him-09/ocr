# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and model
COPY src/ src/
COPY src/best_model.pth src/best_model.pth

# Set the working directory to src
WORKDIR /app/src

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["python", "api.py"] 