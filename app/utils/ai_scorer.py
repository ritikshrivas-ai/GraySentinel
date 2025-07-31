# graysentinel/app/utils/ai_scorer.py
from textblob import TextBlob
import spacy
import json

nlp = spacy.load("en_core_web_sm")

def calculate_risk_score(scan_type, results):
    base_score = 0
    
    # Email scan scoring
    if scan_type == 'email':
        if results.get('breaches'):
            base_score += 30 * len(results['breaches'])
        if results.get('social_media', []):
            base_score += 10
        if results.get('domain_age_days', 0) < 365:
            base_score += 20
    
    # Username scan scoring
    elif scan_type == 'username':
        found_count = len(results.get('found_on', []))
        base_score = min(found_count * 5, 70)
        if found_count == 0:
            base_score += 20
    
    # Social scan scoring
    elif scan_type == 'social':
        activity_score = 0
        for platform, data in results.items():
            if data.get('post_count', 0) > 100:
                activity_score += 10
            if data.get('follower_count', 0) > 1000:
                activity_score += 15
        base_score = min(activity_score, 80)
    
    # Darkweb scan scoring
    elif scan_type == 'darkweb':
        base_score = min(len(results.get('matches', [])) * 15, 100)
    
    # Image scan scoring
    elif scan_type == 'image':
        base_score = 100 if results.get('match_found') else 30
    
    # Sentiment analysis on any text content
    text_content = json.dumps(results)
    doc = nlp(text_content)
    
    # Negative entities boost risk
    negative_entities = ['breach', 'hack', 'leak', 'scam', 'fraud']
    for ent in doc.ents:
        if ent.text.lower() in negative_entities:
            base_score += 15
    
    # Sentiment analysis
    analysis = TextBlob(text_content)
    if analysis.sentiment.polarity < -0.3:
        base_score += 25
    
    # Cap score at 100
    return min(int(base_score), 100)
