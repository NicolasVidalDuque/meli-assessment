FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY v1.py .

# Create output directory
RUN mkdir -p /app/out

# Set matplotlib backend
ENV MPLBACKEND=Agg

ENTRYPOINT ["python", "v1.py"]