# graysentinel/app/utils/async_tasks.py
from celery_app import celery
from app import create_app
from app.core import (
    email_scanner,
    username_scanner,
    social_scanner,
    darkweb_scanner,
    image_matcher
)
from app.models import ScanResult, db
from app.utils.ai_scorer import calculate_risk_score
from app.utils.reporting import generate_pdf_report
import json
import os
import time

app = create_app()

@celery.task(bind=True)
def run_osint_scan(self, scan_type, target, user_id):
    with app.app_context():
        results = {}
        start_time = time.time()
        
        try:
            if scan_type == 'email':
                results = email_scanner.scan(target)
            elif scan_type == 'username':
                results = username_scanner.scan(target)
            elif scan_type == 'social':
                results = social_scanner.scan(target)
            elif scan_type == 'darkweb':
                results = darkweb_scanner.scan(target)
            elif scan_type == 'image':
                results = image_matcher.scan(target)
            else:
                return {'error': 'Invalid scan type'}
            
            # Calculate risk score
            risk_score = calculate_risk_score(scan_type, results)
            
            # Save results to DB
            scan = ScanResult(
                target=target,
                scan_type=scan_type,
                results=json.dumps(results),
                risk_score=risk_score,
                user_id=user_id
            )
            db.session.add(scan)
            db.session.commit()
            
            # Generate PDF report
            report_path = generate_pdf_report(scan.id, target, scan_type, results, risk_score, user_id)
            scan.report_path = report_path
            db.session.commit()
            
            return {
                'status': 'success',
                'results': results,
                'risk_score': risk_score,
                'scan_id': scan.id,
                'time_elapsed': round(time.time() - start_time, 2)
            }
            
        except Exception as e:
            return {'error': str(e)}
