# graysentinel/app/core/email_scanner.py
import holehe
import json

def scan(email):
    results = {
        'email': email,
        'breaches': [],
        'social_media': [],
        'domain_info': {}
    }
    
    # Check for breaches
    breach_data = holehe.breach(email)
    for breach in breach_data:
        if breach['exists']:
            results['breaches'].append({
                'name': breach['name'],
                'domain': breach['domain'],
                'date': breach['breach_date']
            })
    
    # Check social media presence
    social_results = holehe.social(email)
    for site, data in social_results.items():
        if data['exists']:
            results['social_media'].append(site)
    
    # Get domain info
    domain = email.split('@')[-1]
    domain_info = holehe.domain(domain)
    if domain_info:
        results['domain_info'] = {
            'domain': domain,
            'creation_date': domain_info.get('creation_date', ''),
            'expiration_date': domain_info.get('expiration_date', ''),
            'registrar': domain_info.get('registrar', ''),
            'age_days': domain_info.get('age_days', 0)
        }
    
    return results
