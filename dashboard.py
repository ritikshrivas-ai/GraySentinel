#!/usr/bin/env python3
"""
GraySentinel Growth Dashboard
Author: Ritik Shrivas
Purpose: View all growth metrics and data in one place
"""

import json
import datetime
from typing import Dict, Any

def load_json_file(filename: str) -> Dict[str, Any]:
    """Load JSON file and return data"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File {filename} not found")
        return {}
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return {}

def display_growth_blueprint():
    """Display growth blueprint summary"""
    print("🚀 GRAYSENTINEL 60-DAY VIRAL GROWTH BLUEPRINT")
    print("=" * 60)
    
    blueprint = load_json_file("graysentinel_growth_blueprint.json")
    
    if not blueprint:
        return
    
    print(f"📅 Content Pieces: {len(blueprint.get('content_calendar', []))}")
    print(f"💰 Service Tiers: {len(blueprint.get('pricing_structure', {}))}")
    print(f"📊 Projected Monthly Revenue: ₹{blueprint.get('revenue_projection', {}).get('total_monthly_revenue', 0):,}")
    print(f"🤝 Partnership Templates: {len(blueprint.get('partnership_templates', {}))}")
    print(f"📈 Lead Sources: {len(blueprint.get('lead_tracking', {}).get('lead_sources', []))}")
    print(f"🤖 Auto-responder Keywords: {len(blueprint.get('automation_scripts', {}).get('whatsapp_auto_responder', {}).get('keywords', {}))}")
    
    print(f"\n📋 WEEKLY GOALS:")
    for week, goals in blueprint.get('week_goals', {}).items():
        print(f"Week {week}: {goals['leads']} leads, ₹{goals['revenue']:,} revenue, {goals['clients']} clients")

def display_content_calendar():
    """Display content calendar summary"""
    print("\n📱 SOCIAL MEDIA CONTENT CALENDAR")
    print("=" * 40)
    
    calendar = load_json_file("content_calendar.json")
    
    if not calendar:
        return
    
    platforms = {}
    for post in calendar:
        platform = post.get('platform', 'Unknown')
        platforms[platform] = platforms.get(platform, 0) + 1
    
    print(f"📅 Total Posts: {len(calendar)}")
    print(f"📱 Platform Distribution:")
    for platform, count in platforms.items():
        print(f"  {platform}: {count} posts")
    
    print(f"\n📝 SAMPLE CONTENT:")
    for i, post in enumerate(calendar[:3]):
        print(f"\n{i+1}. {post.get('platform', 'Unknown')} - {post.get('content_type', 'Unknown')}")
        print(f"   Content: {post.get('content', '')[:100]}...")
        print(f"   CTA: {post.get('call_to_action', '')}")

def display_content_templates():
    """Display content templates summary"""
    print("\n📝 CONTENT TEMPLATES")
    print("=" * 30)
    
    templates = load_json_file("content_templates.json")
    
    if not templates:
        return
    
    total_templates = sum(len(platform_templates) for platform_templates in templates.values())
    print(f"📝 Total Templates: {total_templates}")
    print(f"📱 Platforms Covered: {', '.join(templates.keys())}")
    
    print(f"\n📋 TEMPLATE BREAKDOWN:")
    for platform, platform_templates in templates.items():
        print(f"  {platform.upper()}: {len(platform_templates)} templates")

def display_revenue_tracker():
    """Display revenue tracker summary"""
    print("\n💰 REVENUE TRACKER")
    print("=" * 25)
    
    tracker_data = load_json_file("revenue_tracker_data.json")
    
    if not tracker_data:
        return
    
    dashboard = tracker_data.get('dashboard_data', {})
    lead_analytics = dashboard.get('lead_analytics', {})
    revenue_analytics = dashboard.get('revenue_analytics', {})
    
    print(f"📊 LEAD ANALYTICS:")
    print(f"  Total Leads: {lead_analytics.get('total_leads', 0)}")
    print(f"  Conversion Rate: {lead_analytics.get('conversion_rate', 0)}%")
    print(f"  Qualified Leads: {lead_analytics.get('qualified_leads', 0)}")
    print(f"  Closed Won: {lead_analytics.get('closed_won', 0)}")
    
    print(f"\n💰 REVENUE ANALYTICS:")
    print(f"  Total Revenue: ₹{revenue_analytics.get('total_revenue', 0):,.2f}")
    print(f"  Monthly Revenue: ₹{revenue_analytics.get('monthly_revenue', 0):,.2f}")
    print(f"  MRR: ₹{revenue_analytics.get('monthly_recurring_revenue', 0):,.2f}")
    print(f"  Avg Deal Size: ₹{revenue_analytics.get('average_deal_size', 0):,.2f}")
    
    print(f"\n📅 WEEKLY PROGRESS:")
    for week_data in dashboard.get('weekly_progress', [])[:4]:
        print(f"  Week {week_data['week']}: {week_data['leads_actual']}/{week_data['leads_target']} leads, "
              f"₹{week_data['revenue_actual']:,.0f}/{week_data['revenue_target']:,.0f} revenue")

def display_quick_actions():
    """Display quick action commands"""
    print("\n🚀 QUICK ACTIONS")
    print("=" * 20)
    
    print("📝 Generate new content calendar:")
    print("   python3 social_media_tools.py")
    
    print("\n💰 Update revenue tracker:")
    print("   python3 revenue_tracker.py")
    
    print("\n📋 Generate content templates:")
    print("   python3 content_templates.py")
    
    print("\n📊 View growth blueprint:")
    print("   python3 growth_blueprint.py")
    
    print("\n📖 Read execution guide:")
    print("   cat EXECUTION_GUIDE.md")

def main():
    """Main dashboard function"""
    print("🛰️ GRAYSENTINEL GROWTH DASHBOARD")
    print("=" * 50)
    print(f"📅 Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Display all sections
    display_growth_blueprint()
    display_content_calendar()
    display_content_templates()
    display_revenue_tracker()
    display_quick_actions()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Review the execution guide (EXECUTION_GUIDE.md)")
    print("2. Start with Week 1 tasks")
    print("3. Use the tools to track progress")
    print("4. Execute consistently for 60 days")
    
    print("\n🚀 Ready to launch your viral growth journey!")
    print("Success is just 60 days away! 💪")

if __name__ == "__main__":
    main()