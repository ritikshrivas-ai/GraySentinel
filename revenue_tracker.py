#!/usr/bin/env python3
"""
GraySentinel Revenue Tracking Dashboard
Author: Ritik Shrivas
Purpose: Track revenue, leads, and growth metrics for the 60-day viral growth plan
"""

import json
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class ServiceTier(Enum):
    FAMILY_PLAN = "family_plan"
    SMB_PLAN = "smb_plan"
    FRAUD_RECOVERY = "fraud_recovery"
    CORPORATE_WORKSHOP = "corporate_workshop"

class LeadStatus(Enum):
    NEW_LEAD = "new_lead"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    PROPOSAL_SENT = "proposal_sent"
    NEGOTIATING = "negotiating"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"
    FOLLOW_UP_REQUIRED = "follow_up_required"

@dataclass
class Lead:
    id: str
    name: str
    contact: str
    source: str
    status: LeadStatus
    service_interest: ServiceTier
    created_date: datetime.datetime
    last_contact: datetime.datetime
    notes: str
    estimated_value: float
    probability: float  # 0-100%

@dataclass
class Revenue:
    id: str
    lead_id: str
    service_tier: ServiceTier
    amount: float
    date: datetime.datetime
    status: str  # "pending", "paid", "cancelled"
    payment_method: str
    notes: str

@dataclass
class WeeklyGoal:
    week: int
    leads_target: int
    revenue_target: float
    clients_target: int
    leads_actual: int
    revenue_actual: float
    clients_actual: int
    completion_percentage: float

class RevenueTracker:
    def __init__(self):
        self.leads = []
        self.revenues = []
        self.weekly_goals = self._initialize_weekly_goals()
        self.service_pricing = self._initialize_pricing()
        
    def _initialize_weekly_goals(self) -> List[WeeklyGoal]:
        """Initialize weekly goals for 60-day plan"""
        
        goals = [
            WeeklyGoal(1, 250, 0, 0, 0, 0, 0, 0),
            WeeklyGoal(2, 500, 25000, 5, 0, 0, 0, 0),
            WeeklyGoal(3, 750, 75000, 15, 0, 0, 0, 0),
            WeeklyGoal(4, 1000, 150000, 25, 0, 0, 0, 0),
            WeeklyGoal(5, 1250, 200000, 40, 0, 0, 0, 0),
            WeeklyGoal(6, 1500, 250000, 50, 0, 0, 0, 0),
            WeeklyGoal(7, 1750, 300000, 75, 0, 0, 0, 0),
            WeeklyGoal(8, 2000, 400000, 100, 0, 0, 0, 0)
        ]
        
        return goals
    
    def _initialize_pricing(self) -> Dict[ServiceTier, Dict[str, Any]]:
        """Initialize service pricing structure"""
        
        return {
            ServiceTier.FAMILY_PLAN: {
                "name": "Family Protection Plan",
                "price": 999,
                "monthly": True,
                "target_clients": 70
            },
            ServiceTier.SMB_PLAN: {
                "name": "Small Business Security",
                "price": 4999,
                "monthly": True,
                "target_clients": 25
            },
            ServiceTier.FRAUD_RECOVERY: {
                "name": "Fraud Recovery Service",
                "price": 2500,
                "monthly": False,
                "target_clients": 15
            },
            ServiceTier.CORPORATE_WORKSHOP: {
                "name": "Corporate Security Workshop",
                "price": 25000,
                "monthly": False,
                "target_clients": 4
            }
        }
    
    def add_lead(self, name: str, contact: str, source: str, service_interest: ServiceTier, 
                 notes: str = "", estimated_value: float = 0, probability: float = 50) -> str:
        """Add a new lead to the system"""
        
        lead_id = f"lead_{len(self.leads) + 1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate estimated value if not provided
        if estimated_value == 0:
            estimated_value = self.service_pricing[service_interest]["price"]
        
        lead = Lead(
            id=lead_id,
            name=name,
            contact=contact,
            source=source,
            status=LeadStatus.NEW_LEAD,
            service_interest=service_interest,
            created_date=datetime.datetime.now(),
            last_contact=datetime.datetime.now(),
            notes=notes,
            estimated_value=estimated_value,
            probability=probability
        )
        
        self.leads.append(lead)
        self._update_weekly_goals()
        
        return lead_id
    
    def update_lead_status(self, lead_id: str, new_status: LeadStatus, notes: str = "") -> bool:
        """Update lead status and add notes"""
        
        for lead in self.leads:
            if lead.id == lead_id:
                lead.status = new_status
                lead.last_contact = datetime.datetime.now()
                if notes:
                    lead.notes += f"\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}: {notes}"
                self._update_weekly_goals()
                return True
        
        return False
    
    def add_revenue(self, lead_id: str, service_tier: ServiceTier, amount: float, 
                   payment_method: str = "UPI", notes: str = "") -> str:
        """Add a revenue entry"""
        
        revenue_id = f"rev_{len(self.revenues) + 1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        revenue = Revenue(
            id=revenue_id,
            lead_id=lead_id,
            service_tier=service_tier,
            amount=amount,
            date=datetime.datetime.now(),
            status="paid",
            payment_method=payment_method,
            notes=notes
        )
        
        self.revenues.append(revenue)
        self._update_weekly_goals()
        
        return revenue_id
    
    def _update_weekly_goals(self):
        """Update weekly goals with actual data"""
        
        current_week = self._get_current_week()
        
        for goal in self.weekly_goals:
            if goal.week == current_week:
                # Count actual leads for this week
                week_start = datetime.datetime.now() - datetime.timedelta(days=7 * (current_week - 1))
                week_end = week_start + datetime.timedelta(days=7)
                
                goal.leads_actual = len([lead for lead in self.leads 
                                       if week_start <= lead.created_date <= week_end])
                
                # Calculate actual revenue for this week
                goal.revenue_actual = sum([rev.amount for rev in self.revenues 
                                         if week_start <= rev.date <= week_end])
                
                # Count actual clients (leads with closed_won status)
                goal.clients_actual = len([lead for lead in self.leads 
                                         if lead.status == LeadStatus.CLOSED_WON and 
                                         week_start <= lead.created_date <= week_end])
                
                # Calculate completion percentage
                if goal.leads_target > 0:
                    goal.completion_percentage = (goal.leads_actual / goal.leads_target) * 100
                else:
                    goal.completion_percentage = 0
    
    def _get_current_week(self) -> int:
        """Get current week number (1-8 for 60-day plan)"""
        
        start_date = datetime.datetime(2024, 1, 1)  # Adjust start date as needed
        current_date = datetime.datetime.now()
        days_passed = (current_date - start_date).days
        return min((days_passed // 7) + 1, 8)
    
    def get_lead_analytics(self) -> Dict[str, Any]:
        """Get comprehensive lead analytics"""
        
        total_leads = len(self.leads)
        new_leads = len([lead for lead in self.leads if lead.status == LeadStatus.NEW_LEAD])
        qualified_leads = len([lead for lead in self.leads if lead.status == LeadStatus.QUALIFIED])
        closed_won = len([lead for lead in self.leads if lead.status == LeadStatus.CLOSED_WON])
        closed_lost = len([lead for lead in self.leads if lead.status == LeadStatus.CLOSED_LOST])
        
        # Lead sources analysis
        sources = {}
        for lead in self.leads:
            sources[lead.source] = sources.get(lead.source, 0) + 1
        
        # Service interest analysis
        service_interest = {}
        for lead in self.leads:
            service = lead.service_interest.value
            service_interest[service] = service_interest.get(service, 0) + 1
        
        # Conversion rates
        conversion_rate = (closed_won / total_leads * 100) if total_leads > 0 else 0
        qualification_rate = (qualified_leads / total_leads * 100) if total_leads > 0 else 0
        
        return {
            "total_leads": total_leads,
            "new_leads": new_leads,
            "qualified_leads": qualified_leads,
            "closed_won": closed_won,
            "closed_lost": closed_lost,
            "conversion_rate": round(conversion_rate, 2),
            "qualification_rate": round(qualification_rate, 2),
            "sources": sources,
            "service_interest": service_interest
        }
    
    def get_revenue_analytics(self) -> Dict[str, Any]:
        """Get comprehensive revenue analytics"""
        
        total_revenue = sum([rev.amount for rev in self.revenues])
        monthly_revenue = sum([rev.amount for rev in self.revenues 
                             if rev.date >= datetime.datetime.now() - datetime.timedelta(days=30)])
        
        # Revenue by service tier
        service_revenue = {}
        for revenue in self.revenues:
            service = revenue.service_tier.value
            service_revenue[service] = service_revenue.get(service, 0) + revenue.amount
        
        # Revenue by payment method
        payment_methods = {}
        for revenue in self.revenues:
            method = revenue.payment_method
            payment_methods[method] = payment_methods.get(method, 0) + revenue.amount
        
        # Monthly recurring revenue (MRR)
        mrr = sum([rev.amount for rev in self.revenues 
                  if self.service_pricing[rev.service_tier]["monthly"]])
        
        return {
            "total_revenue": total_revenue,
            "monthly_revenue": monthly_revenue,
            "monthly_recurring_revenue": mrr,
            "service_revenue": service_revenue,
            "payment_methods": payment_methods,
            "average_deal_size": total_revenue / len(self.revenues) if self.revenues else 0
        }
    
    def get_weekly_progress(self) -> List[Dict[str, Any]]:
        """Get weekly progress report"""
        
        progress = []
        for goal in self.weekly_goals:
            progress.append({
                "week": goal.week,
                "leads_target": goal.leads_target,
                "leads_actual": goal.leads_actual,
                "leads_percentage": round((goal.leads_actual / goal.leads_target * 100) if goal.leads_target > 0 else 0, 2),
                "revenue_target": goal.revenue_target,
                "revenue_actual": goal.revenue_actual,
                "revenue_percentage": round((goal.revenue_actual / goal.revenue_target * 100) if goal.revenue_target > 0 else 0, 2),
                "clients_target": goal.clients_target,
                "clients_actual": goal.clients_actual,
                "clients_percentage": round((goal.clients_actual / goal.clients_target * 100) if goal.clients_target > 0 else 0, 2),
                "overall_completion": round(goal.completion_percentage, 2)
            })
        
        return progress
    
    def get_pipeline_value(self) -> Dict[str, Any]:
        """Get pipeline value and forecasting"""
        
        # Calculate pipeline value by status
        pipeline_value = {}
        for status in LeadStatus:
            leads_with_status = [lead for lead in self.leads if lead.status == status]
            total_value = sum([lead.estimated_value * (lead.probability / 100) for lead in leads_with_status])
            pipeline_value[status.value] = {
                "count": len(leads_with_status),
                "value": total_value
            }
        
        # Calculate weighted pipeline value
        weighted_pipeline = sum([lead.estimated_value * (lead.probability / 100) 
                               for lead in self.leads if lead.status != LeadStatus.CLOSED_WON])
        
        # Forecast next month revenue
        next_month_forecast = weighted_pipeline * 0.3  # 30% of pipeline converts
        
        return {
            "pipeline_value": pipeline_value,
            "weighted_pipeline": weighted_pipeline,
            "next_month_forecast": next_month_forecast
        }
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate complete dashboard data"""
        
        return {
            "lead_analytics": self.get_lead_analytics(),
            "revenue_analytics": self.get_revenue_analytics(),
            "weekly_progress": self.get_weekly_progress(),
            "pipeline_value": self.get_pipeline_value(),
            "current_week": self._get_current_week(),
            "service_pricing": {k.value: v for k, v in self.service_pricing.items()},
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def save_data(self, filename: str = "revenue_tracker_data.json"):
        """Save all data to JSON file"""
        
        # Convert leads to serializable format
        leads_data = []
        for lead in self.leads:
            leads_data.append({
                "id": lead.id,
                "name": lead.name,
                "contact": lead.contact,
                "source": lead.source,
                "status": lead.status.value,
                "service_interest": lead.service_interest.value,
                "created_date": lead.created_date.isoformat(),
                "last_contact": lead.last_contact.isoformat(),
                "notes": lead.notes,
                "estimated_value": lead.estimated_value,
                "probability": lead.probability
            })
        
        # Convert revenues to serializable format
        revenues_data = []
        for revenue in self.revenues:
            revenues_data.append({
                "id": revenue.id,
                "lead_id": revenue.lead_id,
                "service_tier": revenue.service_tier.value,
                "amount": revenue.amount,
                "date": revenue.date.isoformat(),
                "status": revenue.status,
                "payment_method": revenue.payment_method,
                "notes": revenue.notes
            })
        
        # Convert weekly goals to serializable format
        goals_data = []
        for goal in self.weekly_goals:
            goals_data.append({
                "week": goal.week,
                "leads_target": goal.leads_target,
                "revenue_target": goal.revenue_target,
                "clients_target": goal.clients_target,
                "leads_actual": goal.leads_actual,
                "revenue_actual": goal.revenue_actual,
                "clients_actual": goal.clients_actual,
                "completion_percentage": goal.completion_percentage
            })
        
        data = {
            "leads": leads_data,
            "revenues": revenues_data,
            "weekly_goals": goals_data,
            "dashboard_data": self.generate_dashboard_data(),
            "last_saved": datetime.datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return data
    
    def load_data(self, filename: str = "revenue_tracker_data.json"):
        """Load data from JSON file"""
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load leads
            self.leads = []
            for lead_data in data.get("leads", []):
                lead = Lead(
                    id=lead_data["id"],
                    name=lead_data["name"],
                    contact=lead_data["contact"],
                    source=lead_data["source"],
                    status=LeadStatus(lead_data["status"]),
                    service_interest=ServiceTier(lead_data["service_interest"]),
                    created_date=datetime.datetime.fromisoformat(lead_data["created_date"]),
                    last_contact=datetime.datetime.fromisoformat(lead_data["last_contact"]),
                    notes=lead_data["notes"],
                    estimated_value=lead_data["estimated_value"],
                    probability=lead_data["probability"]
                )
                self.leads.append(lead)
            
            # Load revenues
            self.revenues = []
            for revenue_data in data.get("revenues", []):
                revenue = Revenue(
                    id=revenue_data["id"],
                    lead_id=revenue_data["lead_id"],
                    service_tier=ServiceTier(revenue_data["service_tier"]),
                    amount=revenue_data["amount"],
                    date=datetime.datetime.fromisoformat(revenue_data["date"]),
                    status=revenue_data["status"],
                    payment_method=revenue_data["payment_method"],
                    notes=revenue_data["notes"]
                )
                self.revenues.append(revenue)
            
            print(f"✅ Data loaded from {filename}")
            return True
            
        except FileNotFoundError:
            print(f"❌ File {filename} not found. Starting with empty data.")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

def main():
    """Main function to demonstrate revenue tracker"""
    
    print("💰 GraySentinel Revenue Tracker")
    print("=" * 40)
    
    # Initialize tracker
    tracker = RevenueTracker()
    
    # Load existing data if available
    tracker.load_data()
    
    # Add sample leads
    print("📊 Adding sample leads...")
    
    # Sample leads
    sample_leads = [
        ("Rajesh Kumar", "9876543210", "LinkedIn", ServiceTier.FAMILY_PLAN, "Interested in family protection", 999, 70),
        ("Priya Sharma", "9876543211", "Instagram", ServiceTier.SMB_PLAN, "Small business owner", 4999, 60),
        ("Amit Singh", "9876543212", "WhatsApp", ServiceTier.FRAUD_RECOVERY, "Victim of UPI fraud", 2500, 90),
        ("Sneha Patel", "9876543213", "Referral", ServiceTier.CORPORATE_WORKSHOP, "HR Manager", 25000, 40),
        ("Vikram Gupta", "9876543214", "Google My Business", ServiceTier.FAMILY_PLAN, "Looking for cybersecurity", 999, 50)
    ]
    
    for name, contact, source, service, notes, value, prob in sample_leads:
        lead_id = tracker.add_lead(name, contact, source, service, notes, value, prob)
        print(f"✅ Added lead: {name} ({service.value})")
    
    # Add sample revenue
    print("\n💰 Adding sample revenue...")
    
    # Convert some leads to revenue
    for i, lead in enumerate(tracker.leads[:3]):
        if lead.status == LeadStatus.NEW_LEAD:
            tracker.update_lead_status(lead.id, LeadStatus.CLOSED_WON, "Converted to paying client")
            revenue_id = tracker.add_revenue(lead.id, lead.service_interest, lead.estimated_value, "UPI", "Monthly subscription")
            print(f"✅ Added revenue: ₹{lead.estimated_value} from {lead.name}")
    
    # Generate dashboard
    print("\n📈 Generating dashboard...")
    dashboard = tracker.generate_dashboard_data()
    
    # Display key metrics
    print("\n🎯 KEY METRICS:")
    print(f"Total Leads: {dashboard['lead_analytics']['total_leads']}")
    print(f"Conversion Rate: {dashboard['lead_analytics']['conversion_rate']}%")
    print(f"Total Revenue: ₹{dashboard['revenue_analytics']['total_revenue']:,.2f}")
    print(f"Monthly Revenue: ₹{dashboard['revenue_analytics']['monthly_revenue']:,.2f}")
    print(f"Pipeline Value: ₹{dashboard['pipeline_value']['weighted_pipeline']:,.2f}")
    
    # Display weekly progress
    print("\n📅 WEEKLY PROGRESS:")
    for week_data in dashboard['weekly_progress'][:4]:  # Show first 4 weeks
        print(f"Week {week_data['week']}: {week_data['leads_actual']}/{week_data['leads_target']} leads, "
              f"₹{week_data['revenue_actual']:,.0f}/{week_data['revenue_target']:,.0f} revenue")
    
    # Save data
    print("\n💾 Saving data...")
    tracker.save_data()
    print("✅ Data saved to 'revenue_tracker_data.json'")
    
    print("\n🚀 Revenue tracker is ready!")
    print("Use this to track your 60-day viral growth journey! 💪")

if __name__ == "__main__":
    main()