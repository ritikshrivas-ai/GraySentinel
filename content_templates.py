#!/usr/bin/env python3
"""
GraySentinel Content Templates
Author: Ritik Shrivas
Purpose: Generate ready-to-use content templates for all platforms
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ContentTemplate:
    platform: str
    content_type: str
    title: str
    content: str
    hashtags: List[str]
    call_to_action: str
    engagement_boosters: List[str]

class ContentTemplateGenerator:
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
    
    def generate_linkedin_templates(self) -> List[ContentTemplate]:
        """Generate LinkedIn post templates"""
        
        templates = [
            ContentTemplate(
                platform="LinkedIn",
                content_type="post",
                title="Cybersecurity Awareness Post",
                content="""As a cybersecurity expert, I see {number}+ fraud cases daily. The pattern is always the same - urgency, fear, and lack of awareness.

{tip}

That's why I created GraySentinel - to protect families and businesses from these digital predators.

{question} Comment below 👇

#Cybersecurity #DigitalSafety #GraySentinel #Gurugram #FraudPrevention""",
                hashtags=["#Cybersecurity", "#DigitalSafety", "#GraySentinel", "#Gurugram", "#FraudPrevention"],
                call_to_action="Comment your thoughts below",
                engagement_boosters=["Ask questions", "Share personal stories", "Tag relevant people"]
            ),
            
            ContentTemplate(
                platform="LinkedIn",
                content_type="post",
                title="Case Study Post",
                content="""CASE STUDY: {success_story}

The Problem:
- {problem1}
- {problem2}

Our Solution:
✅ {solution1}
✅ {solution2}

The Result:
- {result1}
- {result2}

Ready to protect your business? DM 'BUSINESS' for free security assessment.

#Cybersecurity #CaseStudy #SmallBusiness #GraySentinel #Gurugram""",
                hashtags=["#Cybersecurity", "#CaseStudy", "#SmallBusiness", "#GraySentinel", "#Gurugram"],
                call_to_action="DM 'BUSINESS' for free security assessment",
                engagement_boosters=["Share specific results", "Include numbers", "Ask for engagement"]
            ),
            
            ContentTemplate(
                platform="LinkedIn",
                content_type="post",
                title="Industry Insight Post",
                content="""Why {percentage}% of Indian businesses are vulnerable to cyber attacks:

1. No cybersecurity budget allocation
2. Employee training is often overlooked  
3. Outdated security systems
4. Lack of incident response plans
5. Overconfidence in basic security measures

The reality? Cybercriminals are getting smarter, and your business is a target.

At GraySentinel, we've helped {clients}+ businesses in Gurugram:
- Prevent data breaches worth ₹{amount}+ lakh
- Train {employees}+ employees on cyber safety
- Implement security measures that actually work

{question}

#Cybersecurity #BusinessSecurity #GraySentinel #Gurugram #DigitalTransformation""",
                hashtags=["#Cybersecurity", "#BusinessSecurity", "#GraySentinel", "#Gurugram", "#DigitalTransformation"],
                call_to_action="Share your thoughts",
                engagement_boosters=["Use statistics", "Ask for opinions", "Share industry insights"]
            )
        ]
        
        return templates
    
    def generate_instagram_templates(self) -> List[ContentTemplate]:
        """Generate Instagram Reel templates"""
        
        templates = [
            ContentTemplate(
                platform="Instagram",
                content_type="reel",
                title="Cybersecurity Tips Reel",
                content="""🔥 {hook.upper()} 🔥

{tip1}

{tip2}

BONUS: {tip3}

Swipe up for our FREE cybersecurity checklist 📱

#PhoneHacked #Cybersecurity #DigitalSafety #GraySentinel #TechTips""",
                hashtags=["#PhoneHacked", "#Cybersecurity", "#DigitalSafety", "#GraySentinel", "#TechTips"],
                call_to_action="Swipe up for FREE cybersecurity checklist",
                engagement_boosters=["Use trending audio", "Add text overlays", "Include emojis", "Ask questions"]
            ),
            
            ContentTemplate(
                platform="Instagram",
                content_type="reel",
                title="Success Story Reel",
                content="""🚨 BREAKING: {hook.upper()} 🚨

{success_story}

This is why GraySentinel exists - to protect you!

Comment 'PROTECT' for instant security quote 📱

#Cybersecurity #SuccessStory #GraySentinel #Gurugram #DigitalSafety""",
                hashtags=["#Cybersecurity", "#SuccessStory", "#GraySentinel", "#Gurugram", "#DigitalSafety"],
                call_to_action="Comment 'PROTECT' for instant security quote",
                engagement_boosters=["Use dramatic text", "Include real numbers", "Create urgency"]
            ),
            
            ContentTemplate(
                platform="Instagram",
                content_type="reel",
                title="Warning Reel",
                content="""⚠️ {hook.upper()} ⚠️

{tip1}

{tip2}

Remember: Prevention is better than cure!

DM 'SAFE' for our complete security guide 🔐

#Cybersecurity #DigitalSafety #GraySentinel #TechTips #Security""",
                hashtags=["#Cybersecurity", "#DigitalSafety", "#GraySentinel", "#TechTips", "#Security"],
                call_to_action="DM 'SAFE' for our complete security guide",
                engagement_boosters=["Use warning emojis", "Create urgency", "Include clear CTAs"]
            )
        ]
        
        return templates
    
    def generate_whatsapp_templates(self) -> List[ContentTemplate]:
        """Generate WhatsApp broadcast templates"""
        
        templates = [
            ContentTemplate(
                platform="WhatsApp",
                content_type="broadcast",
                title="Daily Cyber Safety Tip",
                content="""🛡️ GRAYSENTINEL DAILY TIP 🛡️

Today's Focus: {focus}

❌ {tip1}
❌ {tip2}
❌ {tip3}

✅ {tip4}
✅ {tip5}
✅ {tip6}

Remember: {reminder}!

Stay safe, stay protected! 🔐

Reply 'SAFE' for our complete security guide.""",
                hashtags=[],
                call_to_action="Reply 'SAFE' for our complete security guide",
                engagement_boosters=["Personal touch", "Urgency", "Clear call-to-action", "Emojis"]
            ),
            
            ContentTemplate(
                platform="WhatsApp",
                content_type="broadcast",
                title="Urgent Cyber Alert",
                content="""🚨 URGENT CYBER ALERT 🚨

New scam targeting Gurugram residents:

{tip1}

{tip2}

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
                hashtags=[],
                call_to_action="Reply 'ALERT' for real-time fraud updates",
                engagement_boosters=["Create urgency", "Use clear instructions", "Include emojis"]
            ),
            
            ContentTemplate(
                platform="WhatsApp",
                content_type="broadcast",
                title="Success Story Broadcast",
                content="""🎉 SUCCESS STORY 🎉

{success_story}

This is why GraySentinel exists - to protect you and your family!

Our services:
✅ 24/7 Fraud Helpline
✅ Monthly Security Check
✅ Emergency Response
✅ Family Protection Plans

Starting at just ₹33/day!

Reply 'PROTECT' for instant security quote 📱""",
                hashtags=[],
                call_to_action="Reply 'PROTECT' for instant security quote",
                engagement_boosters=["Share success stories", "Include pricing", "Clear CTAs"]
            )
        ]
        
        return templates
    
    def generate_youtube_templates(self) -> List[ContentTemplate]:
        """Generate YouTube video script templates"""
        
        templates = [
            ContentTemplate(
                platform="YouTube",
                content_type="video",
                title="Success Story Video",
                content="""🎥 GRAYSENTINEL SUCCESS STORY 🎥

{hook}

In this video, I'll share:
✅ Real fraud recovery stories
✅ How our 24/7 helpline works
✅ Client testimonials from Gurugram
✅ Our unique approach to cybersecurity

From UPI frauds to business data breaches, we've seen it all and solved it all.

Subscribe to our channel for daily cyber safety tips!

#GraySentinel #Cybersecurity #SuccessStory #FraudPrevention #YouTube""",
                hashtags=["#GraySentinel", "#Cybersecurity", "#SuccessStory", "#FraudPrevention", "#YouTube"],
                call_to_action="Subscribe for daily cyber safety tips",
                engagement_boosters=["Ask for likes", "Encourage comments", "Include timestamps", "Call-to-action"]
            ),
            
            ContentTemplate(
                platform="YouTube",
                content_type="video",
                title="Cybersecurity Tutorial",
                content="""🔥 CYBERSECURITY TUTORIAL 🔥

{hook}

In this video, I'll teach you:
✅ {tip1}
✅ {tip2}
✅ {tip3}
✅ {tip4}

Plus, I'll show you exactly how to implement these security measures!

Don't forget to like and subscribe for more cybersecurity content!

#Cybersecurity #Tutorial #DigitalSafety #GraySentinel #TechTips""",
                hashtags=["#Cybersecurity", "#Tutorial", "#DigitalSafety", "#GraySentinel", "#TechTips"],
                call_to_action="Subscribe for more cybersecurity content",
                engagement_boosters=["Educational content", "Step-by-step instructions", "Visual demonstrations"]
            ),
            
            ContentTemplate(
                platform="YouTube",
                content_type="video",
                title="Live Threat Analysis",
                content="""🚨 LIVE CYBER THREAT ANALYSIS 🚨

{hook}

In this video, I'll analyze:
✅ Latest cyber threats targeting Indians
✅ How to identify and avoid them
✅ What to do if you're a victim
✅ How GraySentinel can help

This is must-watch content for anyone who uses digital payments!

Subscribe for weekly threat analysis!

#Cybersecurity #ThreatAnalysis #DigitalSafety #GraySentinel #LiveAnalysis""",
                hashtags=["#Cybersecurity", "#ThreatAnalysis", "#DigitalSafety", "#GraySentinel", "#LiveAnalysis"],
                call_to_action="Subscribe for weekly threat analysis",
                engagement_boosters=["Live analysis", "Real-time updates", "Expert insights"]
            )
        ]
        
        return templates
    
    def generate_email_templates(self) -> List[ContentTemplate]:
        """Generate email templates for lead nurturing"""
        
        templates = [
            ContentTemplate(
                platform="Email",
                content_type="email",
                title="Welcome Email",
                content="""Subject: Welcome to GraySentinel - Your Digital Safety Partner 🛡️

Hi {name},

Thank you for your interest in GraySentinel cybersecurity services!

I'm Ritik Shrivas, founder of GraySentinel, and I'm excited to help protect you and your family from cyber threats.

As a cybersecurity expert, I've seen firsthand how devastating cyber attacks can be. That's why I created GraySentinel - to provide affordable, accessible cybersecurity solutions for Indian families and businesses.

Here's what you can expect from us:
✅ 24/7 Fraud Helpline
✅ Monthly Security Check
✅ Emergency Response
✅ Family Protection Plans

To get started, I'd love to schedule a free 15-minute security assessment call with you. This will help me understand your specific needs and recommend the best protection plan.

Click here to schedule your free call: [Calendar Link]

In the meantime, here's a free cybersecurity checklist to help you stay safe: [Download Link]

If you have any questions, feel free to reply to this email or call me directly at [Phone Number].

Stay safe and protected!

Best regards,
Ritik Shrivas
Founder, GraySentinel
[Phone] | [Email] | [Website]""",
                hashtags=[],
                call_to_action="Schedule your free security assessment call",
                engagement_boosters=["Personal touch", "Clear value proposition", "Free resources"]
            ),
            
            ContentTemplate(
                platform="Email",
                content_type="email",
                title="Follow-up Email",
                content="""Subject: Quick follow-up on your cybersecurity needs 🔐

Hi {name},

I wanted to follow up on our conversation about your cybersecurity needs.

I understand that {concern} is your biggest concern right now. This is actually very common, and I have good news - it's completely solvable!

At GraySentinel, we've helped {number}+ families and businesses in Gurugram protect themselves from similar threats. Here's what we can do for you:

✅ Immediate threat assessment
✅ Customized protection plan
✅ 24/7 monitoring and support
✅ Emergency response when needed

I'd love to show you exactly how we can help. Would you be available for a quick 15-minute call this week?

Click here to schedule: [Calendar Link]

If you're not ready for a call yet, no problem! Here's a free resource that might help: [Resource Link]

Feel free to reply with any questions you might have.

Best regards,
Ritik Shrivas
Founder, GraySentinel""",
                hashtags=[],
                call_to_action="Schedule a quick 15-minute call",
                engagement_boosters=["Address specific concerns", "Provide social proof", "Offer free resources"]
            ),
            
            ContentTemplate(
                platform="Email",
                content_type="email",
                title="Proposal Email",
                content="""Subject: Your personalized cybersecurity proposal 📋

Hi {name},

Thank you for taking the time to speak with me about your cybersecurity needs.

Based on our conversation, I've prepared a personalized proposal that addresses your specific concerns about {concern}.

Here's what I recommend:

SERVICE: {service_name}
PRICE: ₹{price}/month
FEATURES:
✅ {feature1}
✅ {feature2}
✅ {feature3}
✅ {feature4}

This plan will protect you from {threats} and give you peace of mind knowing that your digital life is secure.

I'm also including a special offer - if you sign up this week, you'll get:
🎁 Free security audit (worth ₹5,000)
🎁 30-day money-back guarantee
🎁 Priority support

To get started, simply reply to this email with "YES" and I'll send you the payment link.

If you have any questions or would like to discuss the proposal, feel free to call me at [Phone Number].

I look forward to helping you stay safe and protected!

Best regards,
Ritik Shrivas
Founder, GraySentinel""",
                hashtags=[],
                call_to_action="Reply with 'YES' to get started",
                engagement_boosters=["Personalized proposal", "Special offers", "Clear next steps"]
            )
        ]
        
        return templates
    
    def generate_all_templates(self) -> Dict[str, List[ContentTemplate]]:
        """Generate all content templates"""
        
        return {
            "linkedin": self.generate_linkedin_templates(),
            "instagram": self.generate_instagram_templates(),
            "whatsapp": self.generate_whatsapp_templates(),
            "youtube": self.generate_youtube_templates(),
            "email": self.generate_email_templates()
        }
    
    def generate_content_variations(self, template: ContentTemplate, count: int = 5) -> List[ContentTemplate]:
        """Generate variations of a template with different content"""
        
        variations = []
        
        for i in range(count):
            # Create variation by replacing placeholders
            content = template.content
            
            # Replace placeholders with random content
            placeholders = {
                "{hook}": random.choice(self.viral_hooks),
                "{tip}": random.choice(self.cybersecurity_tips),
                "{tip1}": random.choice(self.cybersecurity_tips),
                "{tip2}": random.choice(self.cybersecurity_tips),
                "{tip3}": random.choice(self.cybersecurity_tips),
                "{tip4}": random.choice(self.cybersecurity_tips),
                "{tip5}": random.choice(self.cybersecurity_tips),
                "{tip6}": random.choice(self.cybersecurity_tips),
                "{success_story}": random.choice(self.success_stories),
                "{question}": random.choice(self.engagement_questions),
                "{number}": str(random.randint(5, 15)),
                "{percentage}": str(random.randint(60, 80)),
                "{clients}": str(random.randint(50, 100)),
                "{amount}": str(random.randint(1, 10)),
                "{employees}": str(random.randint(200, 500)),
                "{focus}": random.choice(["UPI Security", "Password Safety", "Phishing Prevention", "Social Media Security"]),
                "{reminder}": random.choice(["Banks NEVER ask for your PIN", "Always verify before clicking", "Keep your software updated", "Use strong passwords"]),
                "{problem1}": random.choice(["No cybersecurity measures", "Employee negligence", "Outdated security systems", "Lack of awareness"]),
                "{problem2}": random.choice(["Data breach risk", "UPI fraud vulnerability", "Phishing attacks", "Ransomware threat"]),
                "{solution1}": random.choice(["Comprehensive security audit", "Employee training", "Multi-layer protection", "24/7 monitoring"]),
                "{solution2}": random.choice(["Real-time threat detection", "Automated backups", "Security policies", "Incident response"]),
                "{result1}": random.choice(["Prevented potential loss", "Improved security posture", "Gained customer trust", "Reduced risk by 90%"]),
                "{result2}": random.choice(["Monthly security monitoring", "Ongoing support", "Regular updates", "Peace of mind"])
            }
            
            for placeholder, replacement in placeholders.items():
                content = content.replace(placeholder, replacement)
            
            # Create variation
            variation = ContentTemplate(
                platform=template.platform,
                content_type=template.content_type,
                title=f"{template.title} - Variation {i+1}",
                content=content,
                hashtags=template.hashtags.copy(),
                call_to_action=template.call_to_action,
                engagement_boosters=template.engagement_boosters.copy()
            )
            
            variations.append(variation)
        
        return variations
    
    def save_templates(self, templates: Dict[str, List[ContentTemplate]], filename: str = "content_templates.json"):
        """Save templates to JSON file"""
        
        template_data = {}
        
        for platform, platform_templates in templates.items():
            template_data[platform] = []
            for template in platform_templates:
                template_data[platform].append({
                    "platform": template.platform,
                    "content_type": template.content_type,
                    "title": template.title,
                    "content": template.content,
                    "hashtags": template.hashtags,
                    "call_to_action": template.call_to_action,
                    "engagement_boosters": template.engagement_boosters
                })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)
        
        return template_data

def main():
    """Main function to generate content templates"""
    
    print("📝 GraySentinel Content Template Generator")
    print("=" * 50)
    
    # Initialize template generator
    generator = ContentTemplateGenerator()
    
    # Generate all templates
    print("📝 Generating content templates...")
    templates = generator.generate_all_templates()
    
    total_templates = sum(len(platform_templates) for platform_templates in templates.values())
    print(f"✅ Generated {total_templates} templates across {len(templates)} platforms")
    
    # Generate variations for each template
    print("🔄 Generating template variations...")
    all_templates_with_variations = {}
    
    for platform, platform_templates in templates.items():
        all_templates_with_variations[platform] = []
        
        for template in platform_templates:
            # Add original template
            all_templates_with_variations[platform].append(template)
            
            # Add variations
            variations = generator.generate_content_variations(template, 3)
            all_templates_with_variations[platform].extend(variations)
        
        print(f"✅ Generated {len(all_templates_with_variations[platform])} templates for {platform}")
    
    # Save templates
    print("💾 Saving templates...")
    generator.save_templates(templates)
    print("✅ Templates saved to 'content_templates.json'")
    
    # Display sample templates
    print("\n📋 SAMPLE TEMPLATES:")
    print("-" * 30)
    
    for platform, platform_templates in templates.items():
        print(f"\n{platform.upper()}:")
        for i, template in enumerate(platform_templates[:2]):  # Show first 2 templates
            print(f"  {i+1}. {template.title}")
            print(f"     Content: {template.content[:100]}...")
            print(f"     CTA: {template.call_to_action}")
    
    print(f"\n🎯 TEMPLATE SUMMARY:")
    print(f"📝 Total Templates: {total_templates}")
    print(f"🔄 Variations Generated: {sum(len(platform_templates) for platform_templates in all_templates_with_variations.values())}")
    print(f"📱 Platforms Covered: {', '.join(templates.keys())}")
    
    print("\n🚀 Your content templates are ready!")
    print("Use these to create engaging content across all platforms! 💪")

if __name__ == "__main__":
    main()