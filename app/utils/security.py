# graysentinel/app/utils/security.py
import re

def sanitize_input(input_str):
    """Sanitize user input to prevent XSS and injection attacks"""
    if not input_str:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[;|&$<>`]', '', input_str)
    # Limit length
    return sanitized[:100]

def escape_html(text):
    """Escape HTML special characters"""
    if not text:
        return ""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
