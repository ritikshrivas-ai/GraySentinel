# graysentinel/app/routes.py
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, send_file
from flask_login import login_required, current_user
from .models import ScanResult
from .utils.async_tasks import run_osint_scan
from .utils.reporting import generate_pdf_report
from . import db
import json
import os
import uuid

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/dashboard')
@login_required
def dashboard():
    scans = ScanResult.query.filter_by(user_id=current_user.id).order_by(ScanResult.created_at.desc()).limit(10).all()
    return render_template('dashboard.html', scans=scans)

@main.route('/scan', methods=['POST'])
@login_required
def scan():
    data = request.get_json()
    scan_type = data.get('scan_type')
    target = data.get('target')
    
    if not scan_type or not target:
        return jsonify({'error': 'Missing parameters'}), 400
    
    # Sanitize input
    from .utils.security import sanitize_input
    scan_type = sanitize_input(scan_type)
    target = sanitize_input(target)
    
    # Start async scan
    task = run_osint_scan.delay(scan_type, target, current_user.id)
    
    return jsonify({'task_id': task.id}), 202

@main.route('/report/<int:scan_id>')
@login_required
def download_report(scan_id):
    scan = ScanResult.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        return 'Unauthorized', 403
    
    return send_file(scan.report_path, as_attachment=True)
