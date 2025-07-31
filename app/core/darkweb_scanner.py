# graysentinel/app/core/darkweb_scanner.py
from onionsearch import OnionSearch
import json

def scan(keyword):
    engine = OnionSearch()
    results = engine.search(keyword, max_results=20)
    
    formatted = {
        'keyword': keyword,
        'matches': []
    }
    
    for result in results:
        formatted['matches'].append({
            'title': result.get('title', ''),
            'url': result.get('url', ''),
            'content': result.get('content', '')[:200] + '...' if result.get('content') else '',
            'engine': result.get('engine', '')
        })
    
    return formatted
