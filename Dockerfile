FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip && \
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

# Download and extract external OSINT tools
RUN curl -L https://github.com/WebBreacher/toutatis/archive/refs/heads/master.zip -o toutatis.zip && \
    unzip toutatis.zip && mv toutatis-master /app/toutatis && rm toutatis.zip && \
    curl -L https://github.com/megadose/OnionSearch/archive/refs/heads/master.zip -o onionsearch.zip && \
    unzip onionsearch.zip && mv OnionSearch-master /app/OnionSearch && rm onionsearch.zip && \
    curl -L https://github.com/megadose/Mr.Holmes/archive/refs/heads/master.zip -o mrholmes.zip && \
    unzip mrholmes.zip && mv Mr.Holmes-master /app/MrHolmes && rm mrholmes.zip

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Expose port
EXPOSE 5000

# Run app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:create_app()"]
