# ===============================
# GraySentinel Production Image
# Author: Ritik Shrivas
# ===============================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl unzip build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip and install Python dependencies first (better caching)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        flask \
        flask-login \
        flask-wtf \
        sqlalchemy \
        celery \
        redis \
        fpdf2 \
        imagehash \
        Pillow \
        holehe \
        snscrape \
        instaloader \
        spacy \
        textblob

# Download and extract external OSINT tools
RUN curl -sSLf -o toutatis.zip https://github.com/WebBreacher/toutatis/archive/refs/heads/main.zip && \
    unzip toutatis.zip && mv toutatis-main /app/toutatis && rm toutatis.zip && \
    curl -sSLf -o onionsearch.zip https://github.com/megadose/OnionSearch/archive/refs/heads/main.zip && \
    unzip onionsearch.zip && mv OnionSearch-main /app/OnionSearch && rm onionsearch.zip && \
    curl -sSLf -o mrholmes.zip https://github.com/Lucksi/Mr.Holmes/archive/refs/heads/main.zip && \
    unzip mrholmes.zip && mv Mr.Holmes-main /app/MrHolmes && rm mrholmes.zip

# Install OSINT tool-specific dependencies
RUN pip install --no-cache-dir -r /app/toutatis/requirements.txt && \
    pip install --no-cache-dir -r /app/OnionSearch/requirements.txt && \
    pip install --no-cache-dir -r /app/MrHolmes/requirements.txt || true

# Install Spacy model
RUN python -m spacy download en_core_web_sm

# Security: Create non-root user
RUN useradd -m graysentinel
USER graysentinel

# Expose application port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://redis:6379/0

# Start the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:create_app()"]
