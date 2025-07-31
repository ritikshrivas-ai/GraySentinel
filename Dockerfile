FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        holehe \
        snscrape \
        imagehash \
        Pillow \
        fpdf2 \
        spacy \
        textblob \
        celery \
        redis \
        flask \
        flask-login \
        flask-wtf \
        sqlalchemy

# Clone external OSINT tools
RUN git clone https://github.com/WebBreacher/toutatis /app/toutatis && \
    git clone https://github.com/megadose/OnionSearch /app/OnionSearch && \
    git clone https://github.com/megadose/Mr.Holmes /app/MrHolmes

# Expose port
EXPOSE 5000

# Run app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:create_app()"]
