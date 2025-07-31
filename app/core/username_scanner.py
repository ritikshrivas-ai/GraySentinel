# graysentinel/app/core/username_scanner.py
from maltroller import Maltroller
import json

def scan(username):
    scanner = Maltroller()
    results = scanner.check_username(username)
    
    formatted = {
        'username': username,
        'found_on': [],
        'details': {}
    }
    
    for site, data in results.items():
        if data['exists']:
            formatted['found_on'].append(site)
            formatted['details'][site] = {
                'url': data['url'],
                'response_time': data.get('response_time', 0)
            }
    
    return formatted
