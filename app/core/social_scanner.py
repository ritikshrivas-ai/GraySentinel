# graysentinel/app/core/social_scanner.py
import snscrape.modules.twitter as sntwitter
import instaloader
from toutatis import Toutatis
import json
from datetime import datetime
import time
from flask import current_app

def scan(username):
    results = {}
    
    # Scan Twitter
    try:
        twitter_data = {}
        scraper = sntwitter.TwitterUserScraper(username)
        for i, tweet in enumerate(scraper.get_items()):
            if i >= 50:  # Limit to 50 tweets
                break
            twitter_data.setdefault('tweets', []).append({
                'id': tweet.id,
                'date': tweet.date.isoformat(),
                'content': tweet.content,
                'likes': tweet.likeCount,
                'retweets': tweet.retweetCount
            })
        
        if twitter_data:
            twitter_data['profile_exists'] = True
            results['twitter'] = twitter_data
    except Exception as e:
        results['twitter'] = {'error': str(e)}
    
    # Scan Instagram
    try:
        L = instaloader.Instaloader()
        # Use session ID if available
        if current_app.config['INSTAGRAM_SESSION_ID']:
            L.load_session_from_file(username, current_app.config['INSTAGRAM_SESSION_ID'])
        
        profile = instaloader.Profile.from_username(L.context, username)
        results['instagram'] = {
            'user_id': profile.userid,
            'full_name': profile.full_name,
            'biography': profile.biography,
            'followers': profile.followers,
            'following': profile.followees,
            'post_count': profile.mediacount,
            'is_private': profile.is_private,
            'profile_pic': profile.profile_pic_url
        }
    except Exception as e:
        results['instagram'] = {'error': str(e)}
    
    # Scan Facebook (limited)
    try:
        toutatis = Toutatis()
        fb_data = toutatis.lookup(username)
        if fb_data:
            results['facebook'] = {
                'name': fb_data.get('name'),
                'profile_url': fb_data.get('url'),
                'location': fb_data.get('location'),
                'work': fb_data.get('work')
            }
    except Exception as e:
        results['facebook'] = {'error': str(e)}
    
    return results
