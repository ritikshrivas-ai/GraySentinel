#!/usr/bin/env python3
"""
GraySentinel Social Media Tools
Author: Ritik Shrivas
Purpose: Generate and schedule social media content for viral growth
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SocialMediaPost:
    platform: str
    content_type: str
    content: str
    hashtags: List[str]
    call_to_action: str
    scheduled_time: datetime
    engagement_boosters: List[str]

class SocialMediaContentGenerator:
    def __init__(self):
        self.viral_hooks = [
            "This Gurugram woman lost ₹50,000 to UPI fraud - Here's how she recovered",
            "5 signs your phone is hacked 🔥 (90% Indians don't know #3)",
            "My grandmother almost got scammed - I created GraySentinel to protect others",
            "Small business owners - Your WhatsApp Business account is at risk!",
            "DM 'SAFE' for free cybersecurity checklist",
            "This DLF Phase 4 business saved ₹2 lakh with our security audit",
            "How we helped a Sector 45 family recover stolen UPI money",
            "Gurugram's most affordable cybersecurity - starts at ₹33/day",
            "From 0 to 100 clients - Our Gurugram cybersecurity journey",
            "Why 20 Gurugram businesses trust GraySentinel"
        ]
        
        self.cybersecurity_tips = [
            "Never share your UPI PIN with anyone - banks never ask for it",
            "Enable 2FA on all your important accounts",
            "Check your phone's installed apps regularly for suspicious ones",
            "Use different passwords for different accounts",
            "Always verify the sender before clicking any link",
            "Keep your phone's software updated",
            "Don't use public WiFi for banking or shopping",
            "Enable transaction alerts for all your accounts",
            "Be suspicious of urgent requests for money or information",
            "Keep screenshots of all important transactions"
        ]
        
        self.success_stories = [
            "Prevented ₹2 lakh data breach for DLF Phase 4 manufacturing company",
            "Helped Sector 45 family recover ₹50,000 from UPI fraud",
            "Saved Gurugram startup from ransomware attack",
            "Recovered ₹1.5 lakh for cyber fraud victim in 48 hours",
            "Prevented business email compromise worth ₹5 lakh",
            "Helped elderly couple secure their digital payments",
            "Trained 50+ employees on cyber safety in corporate workshop",
            "Identified and fixed security vulnerability in local business",
            "Recovered stolen social media accounts for influencer",
            "Prevented cryptocurrency wallet hack for investor"
        ]
        
        self.engagement_questions = [
            "What's your biggest cybersecurity concern?",
            "Have you ever been a victim of cyber fraud?",
            "What security measures do you currently use?",
            "Which platform do you feel is most vulnerable?",
            "What would you do if you lost ₹50,000 to fraud?",
            "How often do you check your account statements?",
            "Do you use the same password for multiple accounts?",
            "What's the most suspicious message you've received?",
            "How do you verify if a website is secure?",
            "What's your biggest fear about online banking?"
        ]
    
    def generate_linkedin_post(self, hook: str = None, include_case_study: bool = False) -> SocialMediaPost:
        """Generate a LinkedIn post with professional tone"""
        
        if not hook:
            hook = random.choice(self.viral_hooks)
        
        # Different LinkedIn post templates
        templates = [
            f"""As a cybersecurity expert, I see {random.randint(5, 15)}+ fraud cases daily. The pattern is always the same - urgency, fear, and lack of awareness.

{random.choice(self.cybersecurity_tips)}

That's why I created GraySentinel - to protect families and businesses from these digital predators.

{random.choice(self.engagement_questions)} Comment below 👇

#Cybersecurity #DigitalSafety #GraySentinel #Gurugram #FraudPrevention""",
            
            f"""CASE STUDY: {random.choice(self.success_stories)}

The Problem:
- {random.choice(['No cybersecurity measures', 'Employee negligence', 'Outdated security systems', 'Lack of awareness'])}
- {random.choice(['Data breach risk', 'UPI fraud vulnerability', 'Phishing attacks', 'Ransomware threat'])}

Our Solution:
✅ {random.choice(['Comprehensive security audit', 'Employee training', 'Multi-layer protection', '24/7 monitoring'])}
✅ {random.choice(['Real-time threat detection', 'Automated backups', 'Security policies', 'Incident response'])}

The Result:
- {random.choice(['Prevented potential loss', 'Improved security posture', 'Gained customer trust', 'Reduced risk by 90%'])}
- {random.choice(['Monthly security monitoring', 'Ongoing support', 'Regular updates', 'Peace of mind'])}

Ready to protect your business? DM 'BUSINESS' for free security assessment.

#Cybersecurity #CaseStudy #SmallBusiness #GraySentinel #Gurugram""",
            
            f"""Why {random.randint(60, 80)}% of Indian businesses are vulnerable to cyber attacks:

1. No cybersecurity budget allocation
2. Employee training is often overlooked  
3. Outdated security systems
4. Lack of incident response plans
5. Overconfidence in basic security measures

The reality? Cybercriminals are getting smarter, and your business is a target.

At GraySentinel, we've helped {random.randint(50, 100)}+ businesses in Gurugram:
- Prevent data breaches worth ₹{random.randint(1, 10)}+ lakh
- Train {random.randint(200, 500)}+ employees on cyber safety
- Implement security measures that actually work

{random.choice(self.engagement_questions)}

#Cybersecurity #BusinessSecurity #GraySentinel #Gurugram #DigitalTransformation"""
        ]
        
        content = random.choice(templates)
        hashtags = ["#Cybersecurity", "#DigitalSafety", "#GraySentinel", "#Gurugram", "#FraudPrevention"]
        
        return SocialMediaPost(
            platform="LinkedIn",
            content_type="post",
            content=content,
            hashtags=hashtags,
            call_to_action="Comment your thoughts below",
            scheduled_time=datetime.now() + timedelta(hours=random.randint(1, 24)),
            engagement_boosters=["Ask questions", "Share personal stories", "Tag relevant people"]
        )
    
    def generate_instagram_reel(self, hook: str = None) -> SocialMediaPost:
        """Generate an Instagram Reel with engaging visuals and text"""
        
        if not hook:
            hook = random.choice(self.viral_hooks)
        
        # Instagram Reel templates
        templates = [
            f"""🔥 {hook.upper()} 🔥

{random.choice(self.cybersecurity_tips)}

BONUS: {random.choice(self.cybersecurity_tips)}

Swipe up for our FREE cybersecurity checklist 📱

#PhoneHacked #Cybersecurity #DigitalSafety #GraySentinel #TechTips""",
            
            f"""🚨 BREAKING: {hook.upper()} 🚨

{random.choice(self.success_stories)}

This is why GraySentinel exists - to protect you!

Comment 'PROTECT' for instant security quote 📱

#Cybersecurity #SuccessStory #GraySentinel #Gurugram #DigitalSafety""",
            
            f"""⚠️ {hook.upper()} ⚠️

{random.choice(self.cybersecurity_tips)}

{random.choice(self.cybersecurity_tips)}

Remember: Prevention is better than cure!

DM 'SAFE' for our complete security guide 🔐

#Cybersecurity #DigitalSafety #GraySentinel #TechTips #Security"""
        ]
        
        content = random.choice(templates)
        hashtags = ["#Cybersecurity", "#DigitalSafety", "#GraySentinel", "#TechTips", "#Security"]
        
        return SocialMediaPost(
            platform="Instagram",
            content_type="reel",
            content=content,
            hashtags=hashtags,
            call_to_action="Comment 'PROTECT' for instant quote",
            scheduled_time=datetime.now() + timedelta(hours=random.randint(1, 24)),
            engagement_boosters=["Use trending audio", "Add text overlays", "Include emojis", "Ask questions"]
        )
    
    def generate_whatsapp_broadcast(self, hook: str = None) -> SocialMediaPost:
        """Generate a WhatsApp broadcast message"""
        
        if not hook:
            hook = random.choice(self.viral_hooks)
        
        # WhatsApp broadcast templates
        templates = [
            f"""🛡️ GRAYSENTINEL DAILY TIP 🛡️

Today's Focus: {random.choice(['UPI Security', 'Password Safety', 'Phishing Prevention', 'Social Media Security'])}

❌ {random.choice(self.cybersecurity_tips)}
❌ {random.choice(self.cybersecurity_tips)}
❌ {random.choice(self.cybersecurity_tips)}

✅ {random.choice(self.cybersecurity_tips)}
✅ {random.choice(self.cybersecurity_tips)}
✅ {random.choice(self.cybersecurity_tips)}

Remember: {random.choice(['Banks NEVER ask for your PIN', 'Always verify before clicking', 'Keep your software updated', 'Use strong passwords'])}!

Stay safe, stay protected! 🔐

Reply 'SAFE' for our complete security guide.""",
            
            f"""🚨 URGENT CYBER ALERT 🚨

New scam targeting Gurugram residents:

{random.choice(self.cybersecurity_tips)}

{random.choice(self.cybersecurity_tips)}

If you receive such messages, DO NOT:
❌ Click any links
❌ Share personal information
❌ Enter OTP or PIN
❌ Call back on unknown numbers

Instead:
✅ Report to your bank immediately
✅ Block the number
✅ Share this message with family

Stay vigilant! 🔐

Reply 'ALERT' for real-time fraud updates.""",
            
            f"""🎉 SUCCESS STORY 🎉

{random.choice(self.success_stories)}

This is why GraySentinel exists - to protect you and your family!

Our services:
✅ 24/7 Fraud Helpline
✅ Monthly Security Check
✅ Emergency Response
✅ Family Protection Plans

Starting at just ₹33/day!

Reply 'PROTECT' for instant security quote 📱"""
        ]
        
        content = random.choice(templates)
        hashtags = []
        
        return SocialMediaPost(
            platform="WhatsApp",
            content_type="broadcast",
            content=content,
            hashtags=hashtags,
            call_to_action="Reply with keyword for more info",
            scheduled_time=datetime.now() + timedelta(hours=random.randint(1, 24)),
            engagement_boosters=["Personal touch", "Urgency", "Clear call-to-action", "Emojis"]
        )
    
    def generate_youtube_video_script(self, hook: str = None) -> SocialMediaPost:
        """Generate a YouTube video script"""
        
        if not hook:
            hook = random.choice(self.viral_hooks)
        
        # YouTube video script templates
        templates = [
            f"""🎥 GRAYSENTINEL SUCCESS STORY 🎥

{hook}

In this video, I'll share:
✅ Real fraud recovery stories
✅ How our 24/7 helpline works
✅ Client testimonials from Gurugram
✅ Our unique approach to cybersecurity

From UPI frauds to business data breaches, we've seen it all and solved it all.

Subscribe to our channel for daily cyber safety tips!

#GraySentinel #Cybersecurity #SuccessStory #FraudPrevention #YouTube""",
            
            f"""🔥 CYBERSECURITY TUTORIAL 🔥

{hook}

In this video, I'll teach you:
✅ {random.choice(self.cybersecurity_tips)}
✅ {random.choice(self.cybersecurity_tips)}
✅ {random.choice(self.cybersecurity_tips)}
✅ {random.choice(self.cybersecurity_tips)}

Plus, I'll show you exactly how to implement these security measures!

Don't forget to like and subscribe for more cybersecurity content!

#Cybersecurity #Tutorial #DigitalSafety #GraySentinel #TechTips""",
            
            f"""🚨 LIVE CYBER THREAT ANALYSIS 🚨

{hook}

In this video, I'll analyze:
✅ Latest cyber threats targeting Indians
✅ How to identify and avoid them
✅ What to do if you're a victim
✅ How GraySentinel can help

This is must-watch content for anyone who uses digital payments!

Subscribe for weekly threat analysis!

#Cybersecurity #ThreatAnalysis #DigitalSafety #GraySentinel #LiveAnalysis"""
        ]
        
        content = random.choice(templates)
        hashtags = ["#GraySentinel", "#Cybersecurity", "#DigitalSafety", "#YouTube", "#TechTips"]
        
        return SocialMediaPost(
            platform="YouTube",
            content_type="video",
            content=content,
            hashtags=hashtags,
            call_to_action="Subscribe for daily cyber safety tips",
            scheduled_time=datetime.now() + timedelta(hours=random.randint(1, 24)),
            engagement_boosters=["Ask for likes", "Encourage comments", "Include timestamps", "Call-to-action"]
        )
    
    def generate_content_calendar(self, days: int = 30) -> List[SocialMediaPost]:
        """Generate a content calendar for specified number of days"""
        
        content_calendar = []
        
        for day in range(days):
            # Generate 2-3 posts per day
            posts_per_day = random.randint(2, 3)
            
            for post_num in range(posts_per_day):
                # Randomly select platform
                platform = random.choice(["LinkedIn", "Instagram", "WhatsApp", "YouTube"])
                
                if platform == "LinkedIn":
                    post = self.generate_linkedin_post()
                elif platform == "Instagram":
                    post = self.generate_instagram_reel()
                elif platform == "WhatsApp":
                    post = self.generate_whatsapp_broadcast()
                elif platform == "YouTube":
                    post = self.generate_youtube_video_script()
                
                # Schedule post for the specific day
                post.scheduled_time = datetime.now() + timedelta(days=day, hours=random.randint(9, 21))
                content_calendar.append(post)
        
        return content_calendar
    
    def generate_hashtag_sets(self) -> Dict[str, List[str]]:
        """Generate hashtag sets for different platforms and content types"""
        
        hashtag_sets = {
            "cybersecurity_general": [
                "#Cybersecurity", "#DigitalSafety", "#CyberSecurity", "#InfoSec", "#CyberDefense",
                "#DataProtection", "#Privacy", "#CyberAwareness", "#CyberThreats", "#CyberCrime"
            ],
            
            "indian_cybersecurity": [
                "#CyberSecurityIndia", "#DigitalIndia", "#UPISecurity", "#BankingSecurity",
                "#CyberFraud", "#UPIFraud", "#DigitalPayments", "#CyberSafetyIndia", "#CyberAwarenessIndia"
            ],
            
            "business_security": [
                "#BusinessSecurity", "#SMB", "#SmallBusiness", "#EnterpriseSecurity",
                "#CyberRisk", "#SecurityAudit", "#CyberTraining", "#IncidentResponse", "#CyberCompliance"
            ],
            
            "personal_security": [
                "#PersonalSecurity", "#PhoneSecurity", "#PasswordSecurity", "#SocialMediaSecurity",
                "#OnlineSafety", "#DigitalPrivacy", "#CyberHygiene", "#SafeBrowsing", "#CyberEducation"
            ],
            
            "graysentinel_brand": [
                "#GraySentinel", "#Gurugram", "#CyberExpert", "#CyberConsultant", "#CyberProtection",
                "#CyberHelpline", "#CyberSupport", "#CyberSolutions", "#CyberServices", "#CyberConsulting"
            ],
            
            "viral_engagement": [
                "#Viral", "#Trending", "#MustWatch", "#Important", "#Alert", "#Warning", "#Breaking",
                "#Exclusive", "#Insider", "#ProTip", "#Hack", "#Trick", "#Secret", "#Revealed"
            ]
        }
        
        return hashtag_sets
    
    def generate_engagement_strategies(self) -> Dict[str, List[str]]:
        """Generate engagement strategies for different platforms"""
        
        strategies = {
            "linkedin": [
                "Ask thought-provoking questions",
                "Share personal experiences",
                "Tag relevant industry professionals",
                "Use professional hashtags",
                "Engage with comments within 2 hours",
                "Share industry insights and trends",
                "Post during business hours (9 AM - 5 PM)"
            ],
            
            "instagram": [
                "Use trending audio and music",
                "Add text overlays for key points",
                "Include emojis and visual elements",
                "Post during peak hours (7-9 PM)",
                "Use relevant hashtags (5-10 per post)",
                "Engage with stories and comments",
                "Create carousel posts for detailed content"
            ],
            
            "whatsapp": [
                "Keep messages personal and conversational",
                "Use emojis to make content engaging",
                "Include clear call-to-actions",
                "Send during appropriate hours (9 AM - 8 PM)",
                "Personalize messages with recipient's name",
                "Follow up on responses quickly",
                "Create broadcast lists for different segments"
            ],
            
            "youtube": [
                "Create compelling thumbnails",
                "Write detailed descriptions with timestamps",
                "Use relevant keywords in titles",
                "Post consistently (2-3 times per week)",
                "Engage with comments and community",
                "Create playlists for different topics",
                "Collaborate with other creators"
            ]
        }
        
        return strategies
    
    def save_content_calendar(self, content_calendar: List[SocialMediaPost], filename: str = "content_calendar.json"):
        """Save content calendar to JSON file"""
        
        calendar_data = []
        for post in content_calendar:
            calendar_data.append({
                "platform": post.platform,
                "content_type": post.content_type,
                "content": post.content,
                "hashtags": post.hashtags,
                "call_to_action": post.call_to_action,
                "scheduled_time": post.scheduled_time.isoformat(),
                "engagement_boosters": post.engagement_boosters
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calendar_data, f, indent=2, ensure_ascii=False)
        
        return calendar_data

def main():
    """Main function to generate social media content"""
    
    print("📱 GraySentinel Social Media Content Generator")
    print("=" * 50)
    
    # Initialize content generator
    generator = SocialMediaContentGenerator()
    
    # Generate content calendar
    print("📅 Generating 30-day content calendar...")
    content_calendar = generator.generate_content_calendar(30)
    print(f"✅ Generated {len(content_calendar)} content pieces")
    
    # Generate hashtag sets
    print("🏷️ Generating hashtag sets...")
    hashtag_sets = generator.generate_hashtag_sets()
    print(f"✅ Generated {len(hashtag_sets)} hashtag categories")
    
    # Generate engagement strategies
    print("💡 Generating engagement strategies...")
    strategies = generator.generate_engagement_strategies()
    print(f"✅ Generated strategies for {len(strategies)} platforms")
    
    # Save content calendar
    print("💾 Saving content calendar...")
    generator.save_content_calendar(content_calendar)
    print("✅ Content calendar saved to 'content_calendar.json'")
    
    # Display sample content
    print("\n📝 SAMPLE CONTENT:")
    print("-" * 30)
    
    for i, post in enumerate(content_calendar[:3]):
        print(f"\n{i+1}. {post.platform} - {post.content_type}")
        print(f"Content: {post.content[:100]}...")
        print(f"Hashtags: {', '.join(post.hashtags[:3])}")
        print(f"CTA: {post.call_to_action}")
    
    print(f"\n🎯 CONTENT CALENDAR SUMMARY:")
    print(f"📅 Total Posts: {len(content_calendar)}")
    print(f"🏷️ Hashtag Categories: {len(hashtag_sets)}")
    print(f"💡 Platform Strategies: {len(strategies)}")
    
    print("\n🚀 Your social media content is ready!")
    print("Start posting and watch your engagement grow! 💪")

if __name__ == "__main__":
    main()