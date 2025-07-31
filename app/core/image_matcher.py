# graysentinel/app/core/image_matcher.py
import imagehash
from PIL import Image
import os
import io
import base64
import requests
from app.models import ScanResult
from app import db

def scan(image_data):
    # image_data can be URL or base64 encoded image
    try:
        if image_data.startswith('http'):
            response = requests.get(image_data)
            img = Image.open(io.BytesIO(response.content))
        else:
            # Handle base64 data
            header, data = image_data.split(',', 1)
            img = Image.open(io.BytesIO(base64.b64decode(data)))
        
        # Generate hashes
        ahash = imagehash.average_hash(img)
        dhash = imagehash.dhash(img)
        phash = imagehash.phash(img)
        
        # Compare with previous scans (simplified)
        match_found = False
        # In real implementation, compare with DB of known hashes
        
        return {
            'image_size': img.size,
            'format': img.format,
            'ahash': str(ahash),
            'dhash': str(dhash),
            'phash': str(phash),
            'match_found': match_found
        }
    except Exception as e:
        return {'error': str(e)}
