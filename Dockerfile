	# Use an official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libpoppler-cpp-dev \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the whole project
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK resources
RUN python -c "import nltk; nltk.download('punkt')"


# Expose port
EXPOSE 8000



# Launch FastAPI with uvicorn
CMD ["uvicorn", "frontend.app:app", "--host", "0.0.0.0", "--port", "8000"]
