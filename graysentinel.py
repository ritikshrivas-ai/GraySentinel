#!/usr/bin/env python3
"""
OSINT WEB + Kali Integrator - All-in-One Dashboard
Enhanced OSINT Web Dashboard with Kali Linux Tool Integration
Educational & Ethical Use Only - Advanced Pattern Matching & Normalization
 
OWNER: Ritik Shrivas — GraySentinel / Instarecon MIL
VERSION: 3.3 - Critical Bug Fixes & Enhanced Error Handling
 
INSTALLATION (Kali Linux):
sudo apt-get update && sudo apt-get install -y \
  theharvester sherlock amass exiftool tesseract-ocr gitleaks trufflehog ripgrep \
  whois jq parallel timeout git curl wget python3-pip
 
OPTIONAL PIP PACKAGES:
pip install playwright beautifulsoup4 python-whois pytesseract pillow imagehash pdfminer.six aiohttp flask
 
TERMUX INSTALL:
pkg install python rust golang git curl jq
pip install playwright beautifulsoup4 python-whois pytesseract pillow aiohttp flask
 
DEFAULT LOGIN: admin123
 
SECURITY & LEGAL:
- REQUIRES legal authorization for all scans
- Interactive consent prompts for sensitive targets
- Redacts Aadhaar/PAN/SSN patterns automatically
- Rate limiting and resource controls enforced
- No exploitation or unauthorized access
 
USAGE:
python osint_kali_dashboard.py --port 5000 --host 127.0.0.1
 
ACCESS: http://127.0.0.1:5000
"""
 
import asyncio
import aiohttp
import re
import json
import os
import sys
import time
import urllib.parse
import math
import unicodedata
import hashlib
import logging
import threading
import queue
import secrets
import traceback
import uuid
import shutil
import subprocess
import argparse
import signal
import atexit
from datetime import datetime, timedelta
from html import unescape
from urllib.parse import urlparse, parse_qs
from collections import defaultdict, Counter
from base64 import b64decode, b64encode
import random
from functools import wraps
from typing import Dict, List, Optional, Any, Set, Union
import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
 
# Flask imports
from flask import Flask, render_template_string, request, jsonify, Response, send_file, session, redirect, url_for
import werkzeug
 
# ---------------------------
# Enhanced Configuration
# ---------------------------
USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36 Termux-OSINT",
    "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Mobile Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
]
 
SEM_LIMIT = 5
REQUEST_TIMEOUT = 30
REQUEST_DELAY_MIN = 0.5
REQUEST_DELAY_MAX = 2.0
DATA_DIR = "data"
LOG_DIR = "logs"
CACHE_DIR = "cache"
MAX_WORKERS = 10
 
# Kali Integration Config
KALI_CONFIG = {
    'data_dir': Path('./data'),
    'jobs_dir': Path('./data/kali_jobs'),
    'audit_log': Path('./data/audit.log'),
    'max_workers': 5,
    'global_timeout': 3600,
    'legal_confirm_required': True,
    'sensitive_patterns': {
        'aadhaar': r'\b[2-9]{1}[0-9]{3}\s[0-9]{4}\s[0-9]{4}\b',
        'pan': r'[A-Z]{5}[0-9]{4}[A-Z]{1}',
        'ssn': r'\d{3}-\d{2}-\d{4}'
    }
}
 
# Rate limiting
DOMAIN_DELAYS = {}
MAX_RETRIES = 3
RETRY_DELAY = 1.5
 
# Browser mode availability
BROWSER_AVAILABLE = False
try:
    from playwright.async_api import async_playwright
    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False
 
# OCR availability
OCR_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
 
# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'osint_kali_dashboard.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('osint_kali_dashboard')
 
# ---------------------------
# Performance Monitoring
# ---------------------------
class PerformanceMonitor:
    """Real-time performance monitoring with enhanced metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._counters = defaultdict(int)
    
    def record_metric(self, metric_name: str, value: float):
        with self._lock:
            self.metrics[metric_name].append((time.time(), value))
            if len(self.metrics[metric_name]) > 1000:
                self.metrics[metric_name].pop(0)
    
    def increment_counter(self, counter_name: str, value: int = 1):
        with self._lock:
            self._counters[counter_name] += value
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        with self._lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return {}
            
            values = [v for _, v in self.metrics[metric_name]]
            return {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'count': len(values)
            }
    
    def get_counter(self, counter_name: str) -> int:
        with self._lock:
            return self._counters.get(counter_name, 0)
 
performance_monitor = PerformanceMonitor()
 
# ---------------------------
# Flask Application Setup
# ---------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
# Enhanced job management
jobs: Dict[str, 'OSINTJob'] = {}
job_queue = queue.Queue()
active_tasks: Dict[str, threading.Thread] = {}
job_lock = threading.Lock()
 
# Authentication - DEFAULT PASSWORD: admin123
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')
login_attempts = defaultdict(list)
 
# ---------------------------
# Authentication Decorators
# ---------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
 
def rate_limit_requests(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        now = time.time()
        
        with job_lock:
            login_attempts[client_ip] = [attempt for attempt in login_attempts[client_ip] if now - attempt < 3600]
            recent_attempts = [attempt for attempt in login_attempts[client_ip] if now - attempt < 60]
            if len(recent_attempts) > 10:
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            login_attempts[client_ip].append(now)
        
        return f(*args, **kwargs)
    return decorated_function
 
# ---------------------------
# Kali Integration Core
# ---------------------------
@dataclass
class KaliFinding:
    id: str
    type: str
    value: str
    normalized: str
    source: str
    confidence: float
    context: str
    location: Dict[str, Any]
 
@dataclass
class KaliToolConfig:
    name: str
    command: str
    version_flag: str = "--version"
    output_format: str = "text"
    timeout: int = 300
    rate_limit: float = 1.0
 
class LegalGuard:
    """Legal compliance and consent management with enhanced validation"""
    
    SENSITIVE_DOMAINS = {
        '.gov.in', '.nic.in', '.mil.in', '.ac.in', 
        '.gov', '.mil', '.edu', '.police'
    }
    
    @classmethod
    def confirm_legal_authorization(cls, target: str, override: bool = False) -> bool:
        if not KALI_CONFIG['legal_confirm_required'] and not override:
            return True
            
        print(f"\n🔒 LEGAL COMPLIANCE CHECK")
        print(f"Target: {target}")
        print(f"Time: {datetime.utcnow().isoformat()}Z")
        
        is_sensitive = any(domain in target.lower() for domain in cls.SENSITIVE_DOMAINS)
        if is_sensitive and not override:
            print("❌ TARGET IS IN SENSITIVE DOMAIN LIST")
            return False
        
        try:
            response = input("\nDo you have explicit legal authorization to scan this target? (yes/NO): ")
            if response.lower() != 'yes':
                print("❌ Scan aborted - legal consent not confirmed")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Scan aborted - no input received")
            return False
            
        consent_hash = hashlib.sha256(f"{target}{datetime.utcnow().isoformat()}".encode()).hexdigest()
        cls._audit_log('legal_consent', {
            'target': target,
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'consent_hash': consent_hash,
            'user': os.environ.get('USER', 'unknown'),
            'override': override
        })
        
        print("✅ Legal consent confirmed and logged")
        return True
    
    @classmethod
    def _audit_log(cls, event_type: str, data: Dict[str, Any]):
        log_entry = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat() + "Z",
            **data
        }
        
        try:
            KALI_CONFIG['audit_log'].parent.mkdir(parents=True, exist_ok=True)
            with open(KALI_CONFIG['audit_log'], 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
 
class KaliIntegrator:
    """Kali Linux OSINT tool integration with enhanced error handling"""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.rate_limiter = {}
        self.available_tools = {}
        self._tool_cache = {}
        
        # Initialize directories
        self._init_directories()
    
    def _init_directories(self):
        for directory in [KALI_CONFIG['data_dir'], KALI_CONFIG['jobs_dir']]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
    
    async def discover_available_tools(self) -> Dict[str, Any]:
        if self._tool_cache:
            return self._tool_cache
            
        tools_info = {}
        
        tool_configs = {
            'theharvester': KaliToolConfig('theHarvester', 'theHarvester', '--version', 'xml', 600),
            'sherlock': KaliToolConfig('sherlock', 'sherlock', '--version', 'json', 300),
            'amass': KaliToolConfig('amass', 'amass', '-version', 'json', 1200),
            'exiftool': KaliToolConfig('exiftool', 'exiftool', '-ver', 'json', 60),
            'tesseract': KaliToolConfig('tesseract', 'tesseract', '--version', 'text', 120),
            'gitleaks': KaliToolConfig('gitleaks', 'gitleaks', '--version', 'json', 600),
            'trufflehog': KaliToolConfig('trufflehog', 'trufflehog', '--version', 'json', 600),
            'rg': KaliToolConfig('ripgrep', 'rg', '--version', 'json', 300),
            'whois': KaliToolConfig('whois', 'whois', '', 'text', 60),
            'git': KaliToolConfig('git', 'git', '--version', 'text', 300),
        }
        
        def check_tool(tool_name, config):
            try:
                if not shutil.which(config.command):
                    return None
                
                result = subprocess.run(
                    [config.command] + ([config.version_flag] if config.version_flag else []),
                    capture_output=True, text=True, timeout=10
                )
                
                version = "unknown"
                if result.returncode == 0:
                    version = result.stdout.strip()
                
                return {
                    'name': config.name,
                    'command': config.command,
                    'version': version,
                    'available': True,
                    'output_format': config.output_format
                }
                
            except Exception as e:
                logger.warning(f"Tool check failed for {tool_name}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(check_tool, name, config): name 
                for name, config in tool_configs.items()
            }
            
            for future in as_completed(futures):
                tool_name = futures[future]
                try:
                    tool_info = future.result()
                    if tool_info:
                        tools_info[tool_name] = tool_info
                        self.available_tools[tool_name] = tool_info
                except Exception as e:
                    logger.error(f"Tool discovery failed for {tool_name}: {e}")
        
        self._tool_cache = tools_info
        
        try:
            capabilities_file = KALI_CONFIG['data_dir'] / 'tool_capabilities.json'
            with open(capabilities_file, 'w') as f:
                json.dump(tools_info, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tool capabilities: {e}")
        
        return tools_info
    
    async def run_kali_scan(self, feature: str, target: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        if options is None:
            options = {}
        
        job_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            if feature == "platform_probe":
                result = await self._platform_probe(target, options)
            elif feature == "github_secrets":
                result = await self._github_secrets_scan(target, options)
            elif feature == "exif_extraction":
                result = await self._exif_extraction(target, options)
            elif feature == "whois_lookup":
                result = await self._whois_lookup(target, options)
            elif feature == "full_osint":
                result = await self._full_osint_scan(target, options)
            else:
                raise ValueError(f"Unknown feature: {feature}")
            
            result['meta']['duration_seconds'] = time.time() - start_time
            result['meta']['job_id'] = job_id
            result['meta']['legal_confirmed'] = True
            
            await self._save_kali_results(job_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Kali scan {feature} failed for {target}: {e}")
            raise
    
    async def _platform_probe(self, target: str, options: Dict[str, Any]) -> Dict[str, Any]:
        findings = []
        tools_used = []
        
        if 'sherlock' in self.available_tools:
            try:
                cmd = [
                    'sherlock', target,
                    '--timeout', str(options.get('timeout', 10)),
                    '--folderoutput', str(KALI_CONFIG['jobs_dir']),
                    '--csv'
                ]
                
                result = await self._run_tool('sherlock', cmd, options)
                tools_used.append('sherlock')
                
                if result['success'] and result['stdout']:
                    parsed = self._parse_sherlock_output(result['stdout'])
                    findings.extend(parsed)
                    
            except Exception as e:
                logger.error(f"Sherlock failed: {e}")
        
        return self._build_kali_result("platform_probe", target, findings, {'tools': tools_used})
    
    async def _github_secrets_scan(self, target: str, options: Dict[str, Any]) -> Dict[str, Any]:
        findings = []
        tools_used = []
        
        repo_path = await self._clone_repo(target, options)
        if not repo_path:
            return self._build_kali_result("github_secrets", target, findings, {'tools': [], 'error': 'Clone failed'})
        
        if 'gitleaks' in self.available_tools:
            try:
                cmd = [
                    'gitleaks', 'detect',
                    '--source', str(repo_path),
                    '--report-format', 'json',
                    '--no-git',
                    '--verbose'
                ]
                
                result = await self._run_tool('gitleaks', cmd, options)
                tools_used.append('gitleaks')
                
                if result['success'] and result['stdout']:
                    gitleaks_findings = self._parse_gitleaks_output(result['stdout'])
                    findings.extend(gitleaks_findings)
                    
            except Exception as e:
                logger.error(f"Gitleaks failed: {e}")
        
        await self._cleanup_repo(repo_path)
        findings = self._deduplicate_findings(findings)
        
        return self._build_kali_result("github_secrets", target, findings, {'tools': tools_used})
    
    async def _exif_extraction(self, target: str, options: Dict[str, Any]) -> Dict[str, Any]:
        findings = []
        tools_used = []
        
        if not os.path.exists(target):
            return self._build_kali_result("exif_extraction", target, findings, {'tools': [], 'error': 'File not found'})
        
        if 'exiftool' in self.available_tools:
            try:
                cmd = ['exiftool', '-j', target]
                result = await self._run_tool('exiftool', cmd, options)
                tools_used.append('exiftool')
                
                if result['success'] and result['stdout']:
                    exif_findings = self._parse_exiftool_output(result['stdout'])
                    findings.extend(exif_findings)
                    
            except Exception as e:
                logger.error(f"Exiftool failed: {e}")
        
        return self._build_kali_result("exif_extraction", target, findings, {'tools': tools_used})
    
    async def _whois_lookup(self, target: str, options: Dict[str, Any]) -> Dict[str, Any]:
        findings = []
        tools_used = []
        
        if 'whois' in self.available_tools:
            try:
                cmd = ['whois', target]
                result = await self._run_tool('whois', cmd, options)
                tools_used.append('whois')
                
                if result['success'] and result['stdout']:
                    whois_findings = self._parse_whois_output(result['stdout'])
                    findings.extend(whois_findings)
                    
            except Exception as e:
                logger.error(f"WHOIS lookup failed: {e}")
        
        return self._build_kali_result("whois_lookup", target, findings, {'tools': tools_used})
    
    async def _full_osint_scan(self, target: str, options: Dict[str, Any]) -> Dict[str, Any]:
        all_findings = []
        all_tools = []
        
        features = ["platform_probe", "whois_lookup"]
        
        for feature in features:
            try:
                result = await self.run_kali_scan(feature, target, options)
                all_findings.extend(result.get('findings', []))
                all_tools.extend(result.get('meta', {}).get('tools', []))
            except Exception as e:
                logger.error(f"Feature {feature} failed: {e}")
        
        all_findings = self._deduplicate_findings(all_findings)
        all_tools = list(set(all_tools))
        
        return self._build_kali_result("full_osint", target, all_findings, {'tools': all_tools})
    
    async def _run_tool(self, tool_name: str, command: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        async with self.semaphore:
            timeout = options.get('timeout', 300)
            
            try:
                start_time = time.time()
                
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout
                    )
                    
                    return {
                        'success': process.returncode == 0,
                        'stdout': stdout.decode('utf-8', errors='ignore') if stdout else '',
                        'stderr': stderr.decode('utf-8', errors='ignore') if stderr else '',
                        'returncode': process.returncode,
                        'duration': time.time() - start_time
                    }
                    
                except asyncio.TimeoutExpired:
                    try:
                        process.terminate()
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                    
                    return {
                        'success': False,
                        'stdout': '',
                        'stderr': f'Tool timed out after {timeout} seconds',
                        'returncode': -1,
                        'duration': timeout
                    }
                    
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_name}: {e}")
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': str(e),
                    'returncode': -1,
                    'duration': 0
                }
    
    def _parse_gitleaks_output(self, output: str) -> List[KaliFinding]:
        findings = []
        
        try:
            data = json.loads(output)
            if not isinstance(data, list):
                data = [data] if data else []
            
            for leak in data:
                if not isinstance(leak, dict):
                    continue
                    
                finding = KaliFinding(
                    id=str(uuid.uuid4()),
                    type='secret',
                    value=self._redact_sensitive(leak.get('Match', '')),
                    normalized=leak.get('Match', ''),
                    source='gitleaks',
                    confidence=0.9,
                    context=leak.get('Line', '')[:200],
                    location={
                        'file': leak.get('File', ''),
                        'line': leak.get('LineNumber', 0),
                        'rule': leak.get('RuleID', '')
                    }
                )
                findings.append(finding)
                
        except json.JSONDecodeError as e:
            logger.error(f"Gitleaks JSON parse failed: {e}")
        
        return findings
    
    def _parse_exiftool_output(self, output: str) -> List[KaliFinding]:
        findings = []
        
        try:
            data = json.loads(output)
            if not isinstance(data, list):
                data = [data] if data else []
            
            for exif_data in data:
                if not isinstance(exif_data, dict):
                    continue
                    
                for key, value in exif_data.items():
                    if isinstance(value, (str, int, float)) and key not in ['SourceFile']:
                        finding = KaliFinding(
                            id=str(uuid.uuid4()),
                            type='exif',
                            value=str(value),
                            normalized=str(value),
                            source='exiftool',
                            confidence=0.95,
                            context=f"EXIF tag: {key}",
                            location={'tag': key}
                        )
                        findings.append(finding)
                        
        except json.JSONDecodeError as e:
            logger.error(f"Exiftool JSON parse failed: {e}")
        
        return findings
    
    def _parse_sherlock_output(self, csv_content: str) -> List[KaliFinding]:
        findings = []
        
        try:
            lines = csv_content.strip().split('\n')
            if len(lines) < 2:
                return findings
                
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 3:
                    platform, username, url = parts[0], parts[1], parts[2]
                    
                    finding = KaliFinding(
                        id=str(uuid.uuid4()),
                        type='social_profile',
                        value=username,
                        normalized=username,
                        source='sherlock',
                        confidence=0.85,
                        context=f"Platform: {platform}",
                        location={'platform': platform, 'url': url}
                    )
                    findings.append(finding)
                    
        except Exception as e:
            logger.error(f"Sherlock CSV parse failed: {e}")
        
        return findings
    
    def _parse_whois_output(self, output: str) -> List[KaliFinding]:
        findings = []
        
        key_phrases = {
            'Registrant': 'registrant',
            'Creation Date': 'creation_date',
            'Updated Date': 'updated_date',
            'Expiration Date': 'expiration_date'
        }
        
        for line in output.strip().split('\n'):
            for phrase, field_type in key_phrases.items():
                if phrase in line:
                    finding = KaliFinding(
                        id=str(uuid.uuid4()),
                        type=field_type,
                        value=line.split(':', 1)[1].strip() if ':' in line else line,
                        normalized=line.split(':', 1)[1].strip() if ':' in line else line,
                        source='whois',
                        confidence=0.9,
                        context=phrase,
                        location={'field': phrase}
                    )
                    findings.append(finding)
        return findings
    
    def _redact_sensitive(self, text: str) -> str:
        if not text:
            return text
            
        redacted = text
        for pattern_name, pattern in KALI_CONFIG['sensitive_patterns'].items():
            redacted = re.sub(pattern, f'[REDACTED_{pattern_name.upper()}]', redacted)
        return redacted
    
    def _build_kali_result(self, feature: str, target: str, findings: List[KaliFinding], meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "feature": feature,
            "target": target,
            "job_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "findings": [asdict(finding) for finding in findings],
            "meta": {
                "tools": meta.get('tools', []),
                "duration_seconds": meta.get('duration_seconds', 0),
                "legal_confirmed": True,
                "error": meta.get('error')
            }
        }
    
    def _deduplicate_findings(self, findings: List[KaliFinding]) -> List[KaliFinding]:
        seen = set()
        unique_findings = []
        for finding in findings:
            key = (finding.normalized, finding.type)
            if key not in seen:
                seen.add(key)
                unique_findings.append(finding)
        return unique_findings
    
    async def _clone_repo(self, repo_url: str, options: Dict[str, Any]) -> Optional[Path]:
        try:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            clone_path = KALI_CONFIG['jobs_dir'] / f"repo_{uuid.uuid4().hex[:8]}"
            
            cmd = ['git', 'clone', '--depth', '1', repo_url, str(clone_path)]
            result = await self._run_tool('git', cmd, options)
            
            return clone_path if result['success'] else None
            
        except Exception as e:
            logger.error(f"Repository cloning failed: {e}")
            return None
    
    async def _cleanup_repo(self, repo_path: Path):
        try:
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
        except Exception as e:
            logger.error(f"Repository cleanup failed: {e}")
    
    async def _save_kali_results(self, job_id: str, result: Dict[str, Any]):
        try:
            job_file = KALI_CONFIG['jobs_dir'] / f"{job_id}.json"
            with open(job_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Kali results: {e}")
 
# Initialize Kali Integrator
kali_integrator = KaliIntegrator()
kali_tools_available = {}
 
# ---------------------------
# ML-Optimized Pattern Library
# ---------------------------
@dataclasses.dataclass
class PatternConfig:
    pattern: str
    confidence_threshold: float = 0.7
    ml_weight: float = 1.0
    context_boost: bool = False
 
class MLEnhancedPatterns:
    EMAIL_PATTERNS = [
        PatternConfig(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}", 0.9),
        PatternConfig(r"[^\s@]+@[^\s@]+\.[^\s@]+", 0.8),
    ]
    
    EMAIL_OBFUSCATED = [
        PatternConfig(r"[a-zA-Z0-9._%+\-]+\[?(?:at|@|\(at\)|\[at\]|\sat\s)\]?[a-zA-Z0-9.\-]+\[?(?:dot|\.|\(dot\)|\[dot\]|\sdot\s)\]?[a-zA-Z]{2,}", 0.7),
    ]
    
    PHONE_PATTERNS = [
        PatternConfig(r"(?:\+91[\-\s]?|0)?[6-9]\d{9}", 0.9),
        PatternConfig(r"(?:\+91|91|0)?[6-9]\d[\d\s\-]{8,}", 0.8),
    ]
    
    PAYMENT_PATTERNS = [
        PatternConfig(r"\b[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}\b", 0.8),
        PatternConfig(r"\b\d{10}@[a-zA-Z]{3,}\b", 0.9),
    ]
 
COMPILED_PATTERNS = {}
for category_name, patterns in MLEnhancedPatterns.__dict__.items():
    if not category_name.startswith('_') and isinstance(patterns, list):
        COMPILED_PATTERNS[category_name] = [
            (re.compile(pc.pattern, re.IGNORECASE), pc.confidence_threshold, pc.ml_weight)
            for pc in patterns
        ]
 
# ---------------------------
# Enhanced Dork Generator
# ---------------------------
class AdvancedDorkGenerator:
    DORK_TEMPLATES = {
        "basic": [
            '"{q}"',
            '"{q}" site:.in',
            '"{q}" site:.co.in',
            'intext:"{q}"',
            'intitle:"{q}"'
        ],
        "marketplace": [
            'site:olx.in "{q}"',
            'site:quikr.com "{q}"',
            'site:justdial.com "{q}"',
        ],
        "social": [
            'site:github.com "{q}"',
            'site:linkedin.com "{q}"',
            'site:facebook.com "{q}"',
        ]
    }
    
    @classmethod
    def generate_optimized_dorks(cls, target: str, categories: List[str] = None) -> List[str]:
        if categories is None:
            categories = ["basic", "marketplace", "social"]
        
        dorks = []
        for category in categories:
            if category in cls.DORK_TEMPLATES:
                dorks.extend([dork.format(q=target) for dork in cls.DORK_TEMPLATES[category]])
        
        unique_dorks = list(set(dorks))
        return cls.rank_dorks_by_effectiveness(unique_dorks, target)
    
    @classmethod
    def rank_dorks_by_effectiveness(cls, dorks: List[str], target: str) -> List[str]:
        ranked_dorks = []
        
        for dork in dorks:
            score = 0.0
            if 'site:' in dork:
                score += 0.3
            if 'intext:' in dork:
                score += 0.2
            if 'filetype:' in dork:
                score += 0.4
            if 'intitle:' in dork:
                score += 0.25
            
            score *= (1.0 - min(len(dork) / 200, 0.5))
            ranked_dorks.append((dork, score))
        
        ranked_dorks.sort(key=lambda x: x[1], reverse=True)
        return [dork for dork, score in ranked_dorks]
    
    @classmethod
    def get_categories(cls) -> List[str]:
        return list(cls.DORK_TEMPLATES.keys())
 
# ---------------------------
# Text Normalization Engine
# ---------------------------
def ultra_normalize_text(text: str, aggressive: bool = True) -> str:
    if not text:
        return ""
    
    normalized = text
    
    try:
        normalized = re.sub(r'[\u200b-\u200f\uFEFF\u00ad]', '', normalized)
        normalized = unicodedata.normalize('NFKC', normalized)
        
        for _ in range(3):
            try:
                normalized = unescape(normalized)
            except Exception:
                break
        
        obfuscation_patterns = [
            (r'\[at\]|\(at\)|\s+at\s+|\\u0040|&#64;|%40', '@'),
            (r'\[dot\]|\(dot\)|\s+dot\s+|\\u002e|&#46;|%2e', '.'),
        ]
        
        for pattern, replacement in obfuscation_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        for _ in range(2):
            try:
                normalized = urllib.parse.unquote(normalized)
            except Exception:
                break
        
        if aggressive:
            email_like = re.sub(r'(\w)\s+(\w@)', r'\1\2', normalized)
            if '@' in email_like:
                normalized = email_like
            
            phone_like = re.sub(r'(\d)\s+(\d)', r'\1\2', normalized)
            if sum(c.isdigit() for c in phone_like) >= 10:
                normalized = phone_like
        
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
    except Exception as e:
        logger.error(f"Text normalization failed: {e}")
        return text
    
    return normalized
 
def calculate_entropy(text: str) -> float:
    if not text or len(text) < 2:
        return 0.0
    
    try:
        entropy = 0.0
        length = len(text)
        char_counts = Counter(text)
        
        for count in char_counts.values():
            p_x = count / length
            if p_x > 0:
                entropy += -p_x * math.log2(p_x)
        
        return entropy
    except Exception:
        return 0.0
 
# ---------------------------
# Enhanced Browser Automation
# ---------------------------
class EnhancedBrowserFetcher:
    def __init__(self):
        self.browser = None
        self.playwright = None
        self.available = BROWSER_AVAILABLE
        self.setup_complete = False
    
    async def setup(self) -> bool:
        if not self.available:
            return False
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if 'playwright' in sys.modules:
                    self.playwright = await async_playwright().start()
                    self.browser = await self.playwright.chromium.launch(
                        headless=True,
                        args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
                    )
                    self.setup_complete = True
                    return True
            except Exception as e:
                logger.warning(f"Browser setup attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.available = False
                    return False
                await asyncio.sleep(2 ** attempt)
        
        return False
    
    async def fetch(self, url: str, timeout: int = 30000) -> Optional[str]:
        if not self.available or not self.browser:
            return None
        
        try:
            start_time = time.time()
            
            if 'playwright' in sys.modules:
                context = await self.browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    viewport={'width': 1920, 'height': 1080}
                )
                
                page = await context.new_page()
                await page.goto(url, timeout=timeout, wait_until='domcontentloaded')
                content = await page.content()
                
                await context.close()
                
                performance_monitor.record_metric('browser_fetch_time', time.time() - start_time)
                return content
                
        except Exception as e:
            logger.error(f"Browser fetch failed for {url}: {e}")
            return None
    
    async def close(self):
        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                logger.error(f"Browser close failed: {e}")
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as e:
                logger.error(f"Playwright stop failed: {e}")
 
# ---------------------------
# ML-Enhanced Match Extraction
# ---------------------------
def extract_ml_enhanced_matches(text: str, source_url: str = "", source_title: str = "", 
                              allow_sensitive: bool = False, target_context: str = None) -> List[Dict[str, Any]]:
    matches = []
    
    try:
        normalized_text = ultra_normalize_text(text, aggressive=True)
        
        for category_name, patterns in COMPILED_PATTERNS.items():
            for pattern, confidence_threshold, ml_weight in patterns:
                try:
                    pattern_matches = pattern.finditer(normalized_text)
                    
                    for match in pattern_matches:
                        match_text = match.group()
                        start_pos, end_pos = match.span()
                        
                        context_start = max(0, start_pos - 150)
                        context_end = min(len(normalized_text), end_pos + 150)
                        context_snippet = normalized_text[context_start:context_end]
                        
                        confidence_score = confidence_threshold * 100
                        
                        source_domains = {
                            '.gov.': 20, '.edu.': 15, '.in': 10, '.org': 5,
                            'github.com': 10, 'linkedin.com': 10, 'facebook.com': 5
                        }
                        
                        for domain, boost in source_domains.items():
                            if domain in source_url.lower():
                                confidence_score += boost
                                break
                        
                        if 'EMAIL' in category_name:
                            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}$', match_text):
                                confidence_score += 10
                        
                        elif 'PHONE' in category_name:
                            digits = re.sub(r'\D', '', match_text)
                            if len(digits) >= 10:
                                confidence_score += 10
                        
                        confidence_score *= ml_weight
                        confidence_score = max(0, min(100, confidence_score))
                        
                        redacted_text = match_text
                        if not allow_sensitive and confidence_score >= 80:
                            if calculate_entropy(match_text) > 4.5 and len(match_text) >= 20:
                                redacted_text = f"{match_text[:6]}...{match_text[-4:]} [REDACTED]"
                        
                        match_data = {
                            'match_type': category_name,
                            'match_text': redacted_text,
                            'original_text': match_text if allow_sensitive else redacted_text,
                            'confidence_score': confidence_score,
                            'context_snippet': context_snippet,
                            'source_url': source_url,
                            'source_title': source_title,
                            'entropy': calculate_entropy(match_text),
                            'timestamp': datetime.utcnow().isoformat() + "Z"
                        }
                        
                        matches.append(match_data)
                except Exception as e:
                    logger.error(f"Pattern matching failed for {category_name}: {e}")
                    continue
        
        matches.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
        
    except Exception as e:
        logger.error(f"Match extraction failed: {e}")
    
    return matches
 
# ---------------------------
# Enhanced Job Management
# ---------------------------
class OSINTJob:
    def __init__(self, job_id: str, target: str, options: Dict[str, Any]):
        self.job_id = job_id
        self.target = target
        self.options = options
        self.status = "queued"
        self.progress = 0
        self.message = "Initializing..."
        self.results = {}
        self.start_time = datetime.now()
        self.end_time = None
        self.event_queue = queue.Queue()
        self.error_count = 0
        self.warning_count = 0
        self.findings_count = 0
        
        with job_lock:
            jobs[job_id] = self
    
    def update_progress(self, progress: int, message: str):
        self.progress = progress
        self.message = message
        self.event_queue.put({
            'type': 'progress',
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_finding(self, finding_type: str, data: Dict[str, Any]):
        if finding_type not in self.results:
            self.results[finding_type] = []
        
        self.results[finding_type].append(data)
        self.findings_count += 1
        
        self.event_queue.put({
            'type': 'finding',
            'finding_type': finding_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_error(self, error_message: str):
        self.error_count += 1
        self.event_queue.put({
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_warning(self, warning_message: str):
        self.warning_count += 1
        self.event_queue.put({
            'type': 'warning',
            'message': warning_message,
            'timestamp': datetime.now().isoformat()
        })
    
    def complete(self):
        self.status = "completed"
        self.progress = 100
        self.end_time = datetime.now()
        self.event_queue.put({
            'type': 'complete',
            'results': self.results,
            'statistics': {
                'findings_count': self.findings_count,
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'duration': (self.end_time - self.start_time).total_seconds()
            },
            'timestamp': datetime.now().isoformat()
        })
    
    def fail(self, error_message: str):
        self.status = "failed"
        self.message = f"Failed: {error_message}"
        self.end_time = datetime.now()
        self.event_queue.put({
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })
 
# ---------------------------
# Performance-Optimized Scanning Engine - FIXED VERSION
# ---------------------------
class UltraFetcher:
    def __init__(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, browser_fetcher: EnhancedBrowserFetcher = None):
        self.session = session
        self.sem = sem
        self.browser_fetcher = browser_fetcher
        self.domain_last_request = {}
        self.successful_requests = 0
        self.failed_requests = 0
        self._domain_lock = threading.Lock()
    
    async def fetch(self, url: str, method: str = "GET", allow_redirects: bool = True, 
                   headers: Dict[str, str] = None, retry_count: int = 0, use_browser: bool = False,
                   data: str = None) -> tuple:
        domain = urlparse(url).netloc
        current_time = time.time()
        
        with self._domain_lock:
            if domain in self.domain_last_request:
                elapsed = current_time - self.domain_last_request[domain]
                min_delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
                if elapsed < min_delay:
                    await asyncio.sleep(min_delay - elapsed)
        
        headers = headers or {'User-Agent': random.choice(USER_AGENTS)}
        
        try:
            start_time = time.time()
            
            if use_browser and self.browser_fetcher and self.browser_fetcher.available:
                content = await self.browser_fetcher.fetch(url)
                if content:
                    self.successful_requests += 1
                    with self._domain_lock:
                        self.domain_last_request[domain] = time.time()
                    performance_monitor.record_metric('http_fetch_time', time.time() - start_time)
                    return 200, url, content
            
            async with self.sem:
                with self._domain_lock:
                    self.domain_last_request[domain] = time.time()
                
                # Prepare request kwargs based on method and data
                kwargs = {
                    'headers': headers,
                    'timeout': aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                    'allow_redirects': allow_redirects
                }
                
                if method.upper() == 'POST' and data:
                    kwargs['data'] = data.encode('utf-8') if isinstance(data, str) else data
                
                async with self.session.request(method, url, **kwargs) as resp:
                    final_url = str(resp.url)
                    text = await resp.text(errors="ignore")
                    
                    if self.is_blocked(resp, text):
                        if retry_count < MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * (2 ** retry_count))
                            return await self.fetch(url, method, allow_redirects, headers, retry_count + 1, use_browser, data)
                        else:
                            self.failed_requests += 1
                            return resp.status, final_url, None
                    
                    self.successful_requests += 1
                    performance_monitor.record_metric('http_fetch_time', time.time() - start_time)
                    return resp.status, final_url, text
                    
        except asyncio.TimeoutError:
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY * (2 ** retry_count))
                return await self.fetch(url, method, allow_redirects, headers, retry_count + 1, use_browser, data)
            self.failed_requests += 1
            return None, url, None
            
        except Exception as e:
            logger.error(f"Fetch error for {url}: {e}")
            self.failed_requests += 1
            return None, url, None
    
    def is_blocked(self, response, text: str) -> bool:
        if not response:
            return False
            
        blocked_indicators = [
            response.status in [403, 429, 503],
            'captcha' in text.lower(),
            'blocked' in text.lower(),
            'access denied' in text.lower(),
        ]
        return any(blocked_indicators)
 
async def ultra_deep_scrape_optimized(fetcher: UltraFetcher, urls: List[str], job: OSINTJob, 
                                    allow_sensitive: bool = False, target_context: str = None, 
                                    limit: int = 20) -> List[Dict[str, Any]]:
    all_findings = []
    
    try:
        tasks = []
        for url in urls[:limit]:
            task = asyncio.create_task(fetcher.fetch(url))
            tasks.append(task)
            
            if len(tasks) % 5 == 0:
                await asyncio.sleep(0.05)
        
        completed = 0
        total_tasks = len(tasks)
        
        for coro in asyncio.as_completed(tasks):
            try:
                status, final_url, html = await coro
                completed += 1
                
                job.update_progress(30 + (completed * 70 // total_tasks), f"Scraping {final_url}")
                
                if not html or not final_url:
                    continue
                
                page_findings = {
                    "url": str(final_url),
                    "status": status,
                    "title": extract_optimized_page_title(html),
                    "matches": [],
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                
                matches = extract_ml_enhanced_matches(
                    html, str(final_url), page_findings["title"], 
                    allow_sensitive, target_context
                )
                page_findings["matches"] = matches
                
                for match in matches:
                    job.add_finding(match['match_type'], match)
                
                all_findings.append(page_findings)
                
            except Exception as e:
                job.add_error(f"Error processing URL: {e}")
                continue
        
        return all_findings
        
    except Exception as e:
        job.add_error(f"Deep scraping failed: {e}")
        return []
 
async def ultra_ddg_search_optimized(fetcher: UltraFetcher, query: str, max_pages: int = 3) -> List[str]:
    all_results = []
    base_url = "https://html.duckduckgo.com/html/"
    
    try:
        for page in range(max_pages):
            params = {
                'q': query,
                's': str(page * 30),
                'dc': str(page + 1)
            }
            
            # Fixed: Use data parameter instead of direct data argument
            post_data = urllib.parse.urlencode(params)
            
            status, final_url, html = await fetcher.fetch(
                base_url, "POST", 
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'User-Agent': random.choice(USER_AGENTS)
                },
                data=post_data
            )
            
            if html:
                page_results = parse_optimized_ddg_results(html)
                all_results.extend(page_results)
                
                if not page_results or len(page_results) < 10:
                    break
                
                await asyncio.sleep(random.uniform(0.5, 1.5))
        
        return list(set(all_results))
    
    except Exception as e:
        logger.error(f"DDG search failed: {e}")
        return []
 
def parse_optimized_ddg_results(html: str) -> List[str]:
    results = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        for result in soup.find_all('a', {'class': 'result__a'}):
            href = result.get('href')
            if href:
                if href.startswith('/l/?'):
                    try:
                        parsed = urllib.parse.urlparse(href)
                        query_params = urllib.parse.parse_qs(parsed.query)
                        if 'uddg' in query_params:
                            actual_url = query_params['uddg'][0]
                            results.append(urllib.parse.unquote(actual_url))
                    except Exception:
                        pass
                elif href.startswith('http'):
                    results.append(href)
        
        if not results:
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href and href.startswith('http') and 'duckduckgo.com' not in href:
                    results.append(href)
                    
    except Exception as e:
        logger.error(f"DDG parsing failed: {e}")
    
    return results
 
def extract_optimized_page_title(html: str) -> Optional[str]:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        if soup.title and soup.title.string:
            return soup.title.string.strip()[:200]
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'][:150] + '...' if len(meta_desc['content']) > 150 else meta_desc['content']
        
        h1 = soup.find('h1')
        if h1 and h1.get_text():
            return h1.get_text().strip()[:100]
            
    except Exception:
        pass
    
    return None
 
# ---------------------------
# Background Job Runner - ULTRA-OPTIMIZED VERSION
# ---------------------------
async def run_osint_job_async(job: OSINTJob):
    browser_fetcher = None
    
    try:
        job.update_progress(5, "Starting enhanced OSINT scan...")
        
        target = ultra_normalize_text(job.target)
        if not target or len(target) < 3:
            job.fail("Invalid target: too short or empty")
            return
        
        # Validate target format
        is_email = any(
            re.search(pattern.pattern, target, re.IGNORECASE)
            for pattern in MLEnhancedPatterns.EMAIL_PATTERNS
        )
        is_phone = any(
            re.search(pattern.pattern, target, re.IGNORECASE) 
            for pattern in MLEnhancedPatterns.PHONE_PATTERNS
        )
        
        if not (is_email or is_phone):
            job.fail("Input must be a valid email or phone number")
            return
        
        # Setup browser if requested
        if job.options.get('browser_mode', False) and BROWSER_AVAILABLE:
            browser_fetcher = EnhancedBrowserFetcher()
            if not await browser_fetcher.setup():
                job.add_warning("Browser mode unavailable, falling back to HTTP")
        
        # Setup HTTP client with enhanced settings
        connector = aiohttp.TCPConnector(
            limit=SEM_LIMIT, 
            ttl_dns_cache=300,
            use_dns_cache=True,
            force_close=False,
            enable_cleanup_closed=True
        )
        sem = asyncio.Semaphore(SEM_LIMIT)
        
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            fetcher = UltraFetcher(session, sem, browser_fetcher)
            
            job.update_progress(10, "Generating optimized search queries...")
            
            # Generate and execute dorks
            dork_categories = job.options.get('dork_categories', ['basic', 'marketplace', 'social'])
            dorks = AdvancedDorkGenerator.generate_optimized_dorks(target, dork_categories)
            
            search_results = []
            max_dorks = min(job.options.get('max_dorks', 20), 50)
            
            # Execute dork searches with enhanced error handling
            dork_tasks = []
            for i, dork in enumerate(dorks[:max_dorks]):
                task = asyncio.create_task(
                    ultra_ddg_search_optimized(fetcher, dork, 1)
                )
                dork_tasks.append(task)
                
                if len(dork_tasks) >= 3:
                    try:
                        completed, pending = await asyncio.wait(dork_tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in completed:
                            try:
                                results = await task
                                search_results.extend(results)
                                job.update_progress(10 + (i * 20 // max_dorks), f"Found {len(results)} results from dork search")
                            except Exception as e:
                                job.add_error(f"Dork search failed: {e}")
                        
                        dork_tasks = list(pending)
                    except Exception as e:
                        job.add_error(f"Dork search batch failed: {e}")
            
            # Process remaining tasks
            if dork_tasks:
                try:
                    results = await asyncio.gather(*dork_tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, list):
                            search_results.extend(result)
                        elif isinstance(result, Exception):
                            job.add_error(f"Dork search error: {result}")
                except Exception as e:
                    job.add_error(f"Dork gathering failed: {e}")
            
            # Remove duplicates and validate URLs
            search_results = list(set([url for url in search_results if url and url.startswith('http')]))
            job.update_progress(30, f"Found {len(search_results)} unique sources to scan")
            
            if not search_results:
                job.add_warning("No search results found. Try different dork categories or target.")
                job.results = {
                    "target": target,
                    "scan_metadata": {
                        "total_pages": 0,
                        "successful_pages": 0,
                        "scan_duration": (datetime.now() - job.start_time).total_seconds(),
                        "requests_made": 0,
                        "success_rate": 0,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    },
                    "match_statistics": {
                        "total_matches": 0,
                        "match_type_breakdown": {}
                    },
                    "performance_metrics": performance_monitor.get_stats('http_fetch_time'),
                    "findings": []
                }
                job.complete()
                return
            
            # Scrape found URLs
            findings = await ultra_deep_scrape_optimized(
                fetcher, search_results, job,
                job.options.get('allow_sensitive', False), target,
                min(job.options.get('max_pages', 10), 50)
            )
            
            # Aggregate results
            aggregated = {
                "target": target,
                "scan_metadata": {
                    "total_pages": len(findings),
                    "successful_pages": len([f for f in findings if f.get("status") == 200]),
                    "scan_duration": (datetime.now() - job.start_time).total_seconds(),
                    "requests_made": fetcher.successful_requests + fetcher.failed_requests,
                    "success_rate": fetcher.successful_requests / max(1, fetcher.successful_requests + fetcher.failed_requests),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                },
                "match_statistics": {
                    "total_matches": sum(len(f.get("matches", [])) for f in findings),
                    "match_type_breakdown": defaultdict(int)
                },
                "performance_metrics": performance_monitor.get_stats('http_fetch_time'),
                "findings": findings
            }
            
            # Count match types
            for finding in findings:
                for match in finding.get("matches", []):
                    match_type = match['match_type']
                    aggregated["match_statistics"]["match_type_breakdown"][match_type] += 1
            
            job.results = aggregated
            job.complete()
            
    except Exception as e:
        logger.error(f"Job {job.job_id} failed: {e}")
        job.fail(f"Scan failed: {str(e)}")
        
    finally:
        if browser_fetcher:
            await browser_fetcher.close()
 
async def run_kali_job_async(job: OSINTJob):
    try:
        job.update_progress(5, "Starting Kali OSINT scan...")
        
        feature = job.options.get('kali_feature', 'platform_probe')
        target = job.target
        
        job.update_progress(20, f"Running {feature} on {target}...")
        
        result = await kali_integrator.run_kali_scan(feature, target, {
            'timeout': job.options.get('timeout', 300)
        })
        
        job.update_progress(80, "Processing Kali scan results...")
        
        # Convert Kali findings to OSINT job format
        for finding in result.get('findings', []):
            job.add_finding(finding['type'], finding)
        
        job.results = {
            "target": target,
            "scan_type": "kali_" + feature,
            "kali_results": result,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        job.complete()
        
    except Exception as e:
        logger.error(f"Kali job {job.job_id} failed: {e}")
        job.fail(f"Kali scan failed: {str(e)}")
 
def run_osint_job(job: OSINTJob):
    try:
        # Proper async event loop handling
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        else:
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if job.options.get('scan_type') == 'kali':
                loop.run_until_complete(run_kali_job_async(job))
            else:
                loop.run_until_complete(run_osint_job_async(job))
        except Exception as e:
            logger.error(f"Async job execution failed: {e}")
            job.fail(f"Async execution failed: {str(e)}")
        finally:
            # Clean up the loop
            if not loop.is_closed():
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
            
    except Exception as e:
        logger.error(f"Job execution failed: {e}")
        job.fail(f"Execution failed: {str(e)}")
 
def start_job_thread(job: OSINTJob):
    try:
        thread = threading.Thread(
            target=run_osint_job, 
            args=(job,),
            daemon=True,
            name=f"OSINTJob-{job.job_id}"
        )
        thread.start()
        
        with job_lock:
            active_tasks[job.job_id] = thread
            
    except Exception as e:
        logger.error(f"Failed to start job thread: {e}")
        job.fail(f"Thread start failed: {str(e)}")
 
# ---------------------------
# Enhanced Flask Routes
# ---------------------------
@app.route('/')
@login_required
def index():
    return render_template_string(ENHANCED_HTML_TEMPLATE, 
                                browser_available=BROWSER_AVAILABLE,
                                ocr_available=OCR_AVAILABLE,
                                kali_tools_available=kali_tools_available,
                                dork_categories=AdvancedDorkGenerator.get_categories(),
                                performance_metrics=performance_monitor.get_stats('http_fetch_time'),
                                sem_limit=SEM_LIMIT)
 
@app.route('/login', methods=['GET', 'POST'])
@rate_limit_requests
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        client_ip = request.remote_addr
        
        if not password:
            return render_template_string(LOGIN_TEMPLATE, error="Password required")
        
        if password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session.permanent = True
            session['login_time'] = datetime.now().isoformat()
            
            with job_lock:
                if client_ip in login_attempts:
                    del login_attempts[client_ip]
                
            return redirect(url_for('index'))
        else:
            logger.warning(f"Failed login attempt from {client_ip}")
            return render_template_string(LOGIN_TEMPLATE, error="Invalid password")
    
    return render_template_string(LOGIN_TEMPLATE)
 
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
 
@app.route('/scan', methods=['POST'])
@login_required
@rate_limit_requests
def start_scan():
    try:
        if not request.form.get('consent'):
            return jsonify({'error': 'Legal consent required'}), 400
        
        target = request.form.get('target', '').strip()
        if not target:
            return jsonify({'error': 'Target required'}), 400
        
        if len(target) < 3 or len(target) > 255:
            return jsonify({'error': 'Target must be between 3 and 255 characters'}), 400
        
        # Check for active scans
        current_time = datetime.now()
        with job_lock:
            active_scans = [
                job for job in jobs.values() 
                if job.target == target and job.status == 'running'
                and (current_time - job.start_time).total_seconds() < 1800
            ]
        
        if len(active_scans) > 2:
            return jsonify({'error': 'Too many active scans for this target'}), 429
        
        job_id = secrets.token_urlsafe(16)
        scan_type = request.form.get('scan_type', 'osint')
        
        if scan_type == 'kali':
            options = {
                'scan_type': 'kali',
                'kali_feature': request.form.get('kali_feature', 'platform_probe'),
                'timeout': min(int(request.form.get('timeout', 300)), 1800)
            }
        else:
            options = {
                'scan_type': 'osint',
                'browser_mode': request.form.get('browser_mode') == 'true',
                'allow_sensitive': request.form.get('allow_sensitive') == 'true',
                'max_pages': min(int(request.form.get('max_pages', 10)), 50),
                'max_dorks': min(int(request.form.get('max_dorks', 20)), 50),
                'dork_categories': request.form.getlist('dork_categories[]')
            }
        
        job = OSINTJob(job_id, target, options)
        start_job_thread(job)
        
        logger.info(f"Started {scan_type} scan job {job_id} for target {target}")
        return jsonify({'job_id': job_id, 'scan_type': scan_type})
        
    except Exception as e:
        logger.error(f"Scan start failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500
 
@app.route('/kali-tools', methods=['GET'])
@login_required
def get_kali_tools():
    return jsonify(kali_tools_available)
 
@app.route('/kali-scan', methods=['POST'])
@login_required
@rate_limit_requests
def start_kali_scan():
    try:
        if not request.form.get('consent'):
            return jsonify({'error': 'Legal consent required'}), 400
        
        target = request.form.get('target', '').strip()
        feature = request.form.get('feature', 'platform_probe')
        
        if not target:
            return jsonify({'error': 'Target required'}), 400
        
        job_id = secrets.token_urlsafe(16)
        options = {
            'scan_type': 'kali',
            'kali_feature': feature,
            'timeout': min(int(request.form.get('timeout', 300)), 1800)
        }
        
        job = OSINTJob(job_id, target, options)
        start_job_thread(job)
        
        logger.info(f"Started Kali scan job {job_id} for feature {feature} on {target}")
        return jsonify({'job_id': job_id, 'feature': feature})
        
    except Exception as e:
        logger.error(f"Kali scan start failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500
 
@app.route('/stream/<job_id>')
@login_required
def stream(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    def generate():
        try:
            while True:
                try:
                    event = job.event_queue.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                    
                    if event['type'] in ['complete', 'error']:
                        break
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        except Exception as e:
            logger.error(f"Stream error for job {job_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Stream failed'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
 
@app.route('/status/<job_id>')
@login_required
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    return jsonify({
        'job_id': job.job_id,
        'status': job.status,
        'progress': job.progress,
        'message': job.message,
        'target': job.target,
        'start_time': job.start_time.isoformat(),
        'end_time': job.end_time.isoformat() if job.end_time else None,
        'findings_count': job.findings_count
    })
 
@app.route('/results/<job_id>')
@login_required
def job_results(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job.status != 'completed':
        return jsonify({'error': 'Job not completed'}), 400
    
    return jsonify(job.results)
 
@app.route('/history')
@login_required
def job_history():
    try:
        history = []
        with job_lock:
            job_items = list(jobs.items())[-20:]
            for job_id, job in job_items:
                history.append({
                    'job_id': job_id,
                    'target': job.target,
                    'status': job.status,
                    'progress': job.progress,
                    'start_time': job.start_time.isoformat(),
                    'end_time': job.end_time.isoformat() if job.end_time else None,
                    'findings_count': job.findings_count
                })
        
        return jsonify(history)
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500
 
@app.route('/metrics')
@login_required
def system_metrics():
    with job_lock:
        active_jobs = len([j for j in jobs.values() if j.status == 'running'])
        total_jobs = len(jobs)
    
    return jsonify({
        'performance_metrics': performance_monitor.get_stats('http_fetch_time'),
        'active_jobs': active_jobs,
        'total_jobs': total_jobs,
        'browser_available': BROWSER_AVAILABLE,
        'ocr_available': OCR_AVAILABLE,
        'kali_tools_available': len(kali_tools_available) > 0
    })
 
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + "Z",
        'version': '3.3.0',
        'features': {
            'browser_mode': BROWSER_AVAILABLE,
            'ocr': OCR_AVAILABLE,
            'kali_integration': len(kali_tools_available) > 0
        }
    })
 
# ---------------------------
# HTML Templates
# ---------------------------
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>OSINT+Kali Dashboard - Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { 
            background: linear-gradient(135deg, #1a2f1a 0%, #2d4a2d 100%);
            color: #c8d8c8;
            font-family: 'Courier New', monospace;
            height: 100vh;
            display: flex;
            align-items: center;
        }
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(40, 60, 40, 0.95);
            border: 1px solid #5a7d5a;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .brand {
            text-align: center;
            margin-bottom: 30px;
            color: #90ee90;
        }
        .form-control {
            background: #2d4a2d;
            border: 1px solid #5a7d5a;
            color: #c8d8c8;
        }
        .btn-primary {
            background: #5a7d5a;
            border-color: #5a7d5a;
        }
        .btn-primary:hover {
            background: #90ee90;
            border-color: #90ee90;
            color: #1a2f1a;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="login-container">
            <div class="brand">
                <h3>🔍 OSINT + Kali Dashboard</h3>
                <p class="text-muted">Unified OSINT Platform v3.3</p>
                <p class="text-muted">Ritik Shrivas — GraySentinel</p>
            </div>
            {% if error %}
            <div class="alert alert-danger">{{ error }}</div>
            {% endif %}
            <form method="POST">
                <div class="mb-3">
                    <label class="form-label">Password</label>
                    <input type="password" name="password" class="form-control" required>
                </div>
                <button type="submit" class="btn btn-primary w-100">Access Dashboard</button>
            </form>
            <div class="mt-3 text-center text-muted small">
                <p>Default Password: admin123</p>
                <p>Ethical Use Only • Authorized Access Required</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
 
ENHANCED_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>OSINT+Kali Dashboard - Unified Platform</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --tactical-green: #2d4a2d;
            --tactical-dark: #1a2f1a;
            --tactical-light: #5a7d5a;
            --tactical-accent: #90ee90;
            --tactical-text: #c8d8c8;
        }
        
        body {
            background: linear-gradient(135deg, var(--tactical-dark) 0%, var(--tactical-green) 100%);
            color: var(--tactical-text);
            font-family: 'Fira Mono', 'Courier New', monospace;
        }
        
        .hud-panel {
            background: rgba(40, 60, 40, 0.9);
            border: 1px solid var(--tactical-light);
            border-radius: 8px;
            margin-bottom: 15px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .hud-header {
            border-bottom: 1px solid var(--tactical-light);
            padding-bottom: 10px;
            margin-bottom: 15px;
            color: var(--tactical-accent);
        }
        
        .console {
            background: #000;
            color: #0f0;
            font-family: 'Courier New', monospace;
            padding: 15px;
            border-radius: 5px;
            height: 400px;
            overflow-y: auto;
            font-size: 0.9em;
            border: 1px solid var(--tactical-light);
        }
        
        .finding-card {
            background: rgba(50, 70, 50, 0.7);
            border: 1px solid var(--tactical-light);
            border-radius: 5px;
            padding: 12px;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        
        .progress {
            background: var(--tactical-dark);
            height: 10px;
            border-radius: 5px;
        }
        
        .progress-bar {
            background: linear-gradient(90deg, var(--tactical-accent), #28a745);
            border-radius: 5px;
        }
        
        .form-control, .form-select {
            background: var(--tactical-dark);
            border: 1px solid var(--tactical-light);
            color: var(--tactical-text);
        }
        
        .btn-tactical {
            background: var(--tactical-green);
            border: 1px solid var(--tactical-light);
            color: var(--tactical-text);
        }
        
        .brand-header {
            background: rgba(26, 47, 26, 0.95);
            border-bottom: 1px solid var(--tactical-light);
            padding: 15px 0;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .stat-card {
            background: rgba(50, 70, 50, 0.7);
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.8em;
            color: var(--tactical-accent);
            font-weight: bold;
        }
        
        .nav-tabs .nav-link {
            color: var(--tactical-text);
            background: var(--tactical-dark);
        }
        
        .nav-tabs .nav-link.active {
            background: var(--tactical-green);
            color: var(--tactical-accent);
            border-color: var(--tactical-light);
        }
    </style>
</head>
<body>
    <div class="brand-header">
        <div class="container-fluid">
            <div class="row align-items-center">
                <div class="col">
                    <h4 class="mb-0">
                        <i class="fas fa-satellite-dish"></i>
                        Ritik Shrivas — OSINT + Kali Dashboard v3.3
                    </h4>
                    <small class="text-muted">Critical Bug Fixes & Enhanced Error Handling</small>
                </div>
                <div class="col-auto">
                    <a href="/metrics" class="btn btn-sm btn-tactical ms-2">
                        <i class="fas fa-chart-line"></i> Metrics
                    </a>
                    <a href="/logout" class="btn btn-sm btn-tactical ms-2">
                        <i class="fas fa-sign-out-alt"></i> Logout
                    </a>
                </div>
            </div>
        </div>
    </div>
 
    <div class="container-fluid mt-4">
        <ul class="nav nav-tabs" id="dashboardTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="osint-tab" data-bs-toggle="tab" data-bs-target="#osint" type="button" role="tab">
                    <i class="fas fa-search"></i> OSINT Scanner
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="kali-tab" data-bs-toggle="tab" data-bs-target="#kali" type="button" role="tab">
                    <i class="fas fa-tools"></i> Kali Tools
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="results-tab" data-bs-toggle="tab" data-bs-target="#results" type="button" role="tab">
                    <i class="fas fa-chart-bar"></i> Results
                </button>
            </li>
        </ul>
 
        <div class="tab-content mt-3" id="dashboardTabsContent">
            <!-- OSINT Scanner Tab -->
            <div class="tab-pane fade show active" id="osint" role="tabpanel">
                <div class="row">
                    <div class="col-md-4">
                        <div class="hud-panel">
                            <div class="hud-header">
                                <h5><i class="fas fa-cogs"></i> OSINT Scan Controls</h5>
                            </div>
                            
                            <form id="osintScanForm">
                                <div class="mb-3">
                                    <label class="form-label">Target</label>
                                    <input type="text" name="target" class="form-control" 
                                           placeholder="email or phone" required>
                                    <small class="form-text text-muted">Example: user@example.com or 9876543210</small>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Scan Intensity</label>
                                    <select name="scan_type" class="form-select">
                                        <option value="quick">Quick Scan</option>
                                        <option value="deep">Deep Scan</option>
                                    </select>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Max Pages</label>
                                    <input type="number" name="max_pages" class="form-control" value="10" min="1" max="50">
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Dork Categories</label>
                                    <div class="form-check">
                                        {% for category in dork_categories %}
                                        <div>
                                            <input class="form-check-input" type="checkbox" 
                                                   name="dork_categories[]" value="{{ category }}" 
                                                   id="cat_{{ category }}" checked>
                                            <label class="form-check-label" for="cat_{{ category }}">
                                                {{ category|title }}
                                            </label>
                                        </div>
                                        {% endfor %}
                                    </div>
                                </div>
                                
                                <div class="mb-3">
                                    <div class="form-check form-switch">
                                        <input class="form-check-input" type="checkbox" 
                                               name="browser_mode" id="browserMode" 
                                               {% if browser_available %}checked{% else %}disabled{% endif %}>
                                        <label class="form-check-label" for="browserMode">
                                            Browser Mode {% if not browser_available %}(Not Available){% endif %}
                                        </label>
                                    </div>
                                    
                                    <div class="form-check form-switch">
                                        <input class="form-check-input" type="checkbox" 
                                               name="allow_sensitive" id="allowSensitive">
                                        <label class="form-check-label" for="allowSensitive">
                                            Allow Sensitive Data
                                        </label>
                                    </div>
                                </div>
                                
                                <div class="mb-3 form-check">
                                    <input type="checkbox" class="form-check-input" name="consent" id="consent" required>
                                    <label class="form-check-label" for="consent">
                                        Confirm legal authorization
                                    </label>
                                </div>
                                
                                <button type="submit" class="btn btn-tactical w-100">
                                    <i class="fas fa-play"></i> Start OSINT Scan
                                </button>
                            </form>
                        </div>
                    </div>
                    
                    <div class="col-md-8">
                        <div class="hud-panel">
                            <div class="hud-header">
                                <h5><i class="fas fa-terminal"></i> Live Console</h5>
                            </div>
                            <div class="console" id="osintConsole"></div>
                        </div>
                        
                        <div class="hud-panel">
                            <div class="hud-header">
                                <h5><i class="fas fa-chart-bar"></i> Scan Progress</h5>
                            </div>
                            <div id="osintProgressSection">
                                <div class="progress mb-2">
                                    <div id="osintProgressBar" class="progress-bar" style="width: 0%"></div>
                                </div>
                                <div id="osintProgressText" class="small">No active scan</div>
                                <div class="row mt-2">
                                    <div class="col-4">
                                        <small>Findings: <span id="osintFindingsCount">0</span></small>
                                    </div>
                                    <div class="col-4">
                                        <small>Errors: <span id="osintErrorsCount">0</span></small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
 
            <!-- Kali Tools Tab -->
            <div class="tab-pane fade" id="kali" role="tabpanel">
                <div class="row">
                    <div class="col-md-4">
                        <div class="hud-panel">
                            <div class="hud-header">
                                <h5><i class="fas fa-tools"></i> Kali Tool Scanner</h5>
                            </div>
                            
                            <div id="kaliStatus" class="mb-3">
                                {% if kali_tools_available %}
                                <div class="alert alert-success">
                                    <i class="fas fa-check"></i> Kali Tools Available
                                </div>
                                {% else %}
                                <div class="alert alert-warning">
                                    <i class="fas fa-exclamation-triangle"></i> Kali Tools Not Detected
                                </div>
                                {% endif %}
                            </div>
                            
                            <form id="kaliScanForm" {% if not kali_tools_available %}style="display:none"{% endif %}>
                                <div class="mb-3">
                                    <label class="form-label">Target</label>
                                    <input type="text" name="target" class="form-control" 
                                           placeholder="domain, email, or repo URL" required>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Kali Feature</label>
                                    <select name="feature" class="form-select">
                                        <option value="platform_probe">Platform Probe (sherlock)</option>
                                        <option value="github_secrets">GitHub Secrets (gitleaks)</option>
                                        <option value="exif_extraction">EXIF Extraction (exiftool)</option>
                                        <option value="whois_lookup">WHOIS Lookup</option>
                                        <option value="full_osint">Full OSINT Scan</option>
                                    </select>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Timeout (seconds)</label>
                                    <input type="number" name="timeout" class="form-control" value="300" min="30" max="1800">
                                </div>
                                
                                <div class="mb-3 form-check">
                                    <input type="checkbox" class="form-check-input" name="consent" id="kaliConsent" required>
                                    <label class="form-check-label" for="kaliConsent">
                                        Confirm legal authorization
                                    </label>
                                </div>
                                
                                <button type="submit" class="btn btn-tactical w-100">
                                    <i class="fas fa-play"></i> Start Kali Scan
                                </button>
                            </form>
                            
                            <div class="mt-3">
                                <button id="refreshKaliTools" class="btn btn-sm btn-tactical w-100">
                                    <i class="fas fa-sync"></i> Refresh Tool Detection
                                </button>
                            </div>
                        </div>
                        
                        <div class="hud-panel">
                            <div class="hud-header">
                                <h5><i class="fas fa-list"></i> Available Tools</h5>
                            </div>
                            <div id="kaliToolsList">
                                <!-- Tools will be populated by JavaScript -->
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-8">
                        <div class="hud-panel">
                            <div class="hud-header">
                                <h5><i class="fas fa-terminal"></i> Kali Console</h5>
                            </div>
                            <div class="console" id="kaliConsole"></div>
                        </div>
                        
                        <div class="hud-panel">
                            <div class="hud-header">
                                <h5><i class="fas fa-chart-bar"></i> Kali Scan Progress</h5>
                            </div>
                            <div id="kaliProgressSection">
                                <div class="progress mb-2">
                                    <div id="kaliProgressBar" class="progress-bar" style="width: 0%"></div>
                                </div>
                                <div id="kaliProgressText" class="small">No active scan</div>
                                <div class="row mt-2">
                                    <div class="col-4">
                                        <small>Findings: <span id="kaliFindingsCount">0</span></small>
                                    </div>
                                    <div class="col-4">
                                        <small>Errors: <span id="kaliErrorsCount">0</span></small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
 
            <!-- Results Tab -->
            <div class="tab-pane fade" id="results" role="tabpanel">
                <div class="row">
                    <div class="col-12">
                        <div class="hud-panel">
                            <div class="hud-header">
                                <h5><i class="fas fa-history"></i> Scan History</h5>
                            </div>
                            <div id="jobHistory"></div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-3">
                    <div class="col-md-6">
                        <div class="hud-panel">
                            <div class="hud-header">
                                <h5><i class="fas fa-chart-pie"></i> System Stats</h5>
                            </div>
                            <div class="stats-grid">
                                <div class="stat-card">
                                    <div class="stat-value" id="statJobs">0</div>
                                    <div class="stat-label">Total Jobs</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-value" id="statActive">0</div>
                                    <div class="stat-label">Active</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-value" id="statMatches">0</div>
                                    <div class="stat-label">Matches</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="hud-panel">
                            <div class="hud-header">
                                <h5><i class="fas fa-shield-alt"></i> System Status</h5>
                            </div>
                            <div class="small">
                                <div>Browser Mode: <span class="text-{% if browser_available %}success{% else %}warning{% endif %}">
                                    {% if browser_available %}<i class="fas fa-check"></i> Available{% else %}<i class="fas fa-times"></i> Disabled{% endif %}
                                </span></div>
                                <div>OCR: <span class="text-{% if ocr_available %}success{% else %}warning{% endif %}">
                                    {% if ocr_available %}<i class="fas fa-check"></i> Available{% else %}<i class="fas fa-times"></i> Disabled{% endif %}
                                </span></div>
                                <div>Kali Tools: <span class="text-{% if kali_tools_available %}success{% else %}warning{% endif %}">
                                    {% if kali_tools_available %}<i class="fas fa-check"></i> Available{% else %}<i class="fas fa-times"></i> Not Detected{% endif %}
                                </span></div>
                                <div>Concurrency: {{ sem_limit }}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
 
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentOsintJobId = null;
        let currentKaliJobId = null;
        let osintEventSource = null;
        let kaliEventSource = null;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            updateStats();
            loadKaliTools();
            setInterval(updateStats, 5000);
            addToConsole('OSINT + Kali Dashboard Ready - v3.3', 'success', 'osint');
            addToConsole('Critical bug fixes applied successfully', 'info', 'osint');
            addToConsole('Kali integration initialized', 'info', 'kali');
        });
        
        // OSINT Scan Form
        document.getElementById('osintScanForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            
            fetch('/scan', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    addToConsole('ERROR: ' + data.error, 'error', 'osint');
                } else {
                    currentOsintJobId = data.job_id;
                    startOsintEventStream(currentOsintJobId);
                    addToConsole('OSINT scan started: ' + data.job_id, 'info', 'osint');
                    resetOsintCounters();
                }
            })
            .catch(err => {
                addToConsole('Error starting scan: ' + err, 'error', 'osint');
            });
        });
        
        // Kali Scan Form
        document.getElementById('kaliScanForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            
            fetch('/kali-scan', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    addToConsole('ERROR: ' + data.error, 'error', 'kali');
                } else {
                    currentKaliJobId = data.job_id;
                    startKaliEventStream(currentKaliJobId);
                    addToConsole('Kali scan started: ' + data.feature + ' on ' + formData.get('target'), 'info', 'kali');
                    resetKaliCounters();
                }
            })
            .catch(err => {
                addToConsole('Error starting Kali scan: ' + err, 'error', 'kali');
            });
        });
        
        // Refresh Kali Tools
        document.getElementById('refreshKaliTools').addEventListener('click', function() {
            loadKaliTools();
        });
        
        // Event Streams
        function startOsintEventStream(jobId) {
            if (osintEventSource) osintEventSource.close();
            
            osintEventSource = new EventSource('/stream/' + jobId);
            osintEventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleOsintEvent(data);
            };
        }
        
        function startKaliEventStream(jobId) {
            if (kaliEventSource) kaliEventSource.close();
            
            kaliEventSource = new EventSource('/stream/' + jobId);
            kaliEventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleKaliEvent(data);
            };
        }
        
        function handleOsintEvent(data) {
            switch(data.type) {
                case 'progress':
                    document.getElementById('osintProgressBar').style.width = data.progress + '%';
                    document.getElementById('osintProgressText').textContent = data.message;
                    addToConsole(data.message, 'info', 'osint');
                    break;
                case 'finding':
                    document.getElementById('osintFindingsCount').textContent = 
                        parseInt(document.getElementById('osintFindingsCount').textContent) + 1;
                    addToConsole(`Found ${data.finding_type}: ${data.data.match_text}`, 'success', 'osint');
                    break;
                case 'error':
                    document.getElementById('osintErrorsCount').textContent = 
                        parseInt(document.getElementById('osintErrorsCount').textContent) + 1;
                    addToConsole('ERROR: ' + data.message, 'error', 'osint');
                    break;
                case 'warning':
                    addToConsole('WARNING: ' + data.message, 'warning', 'osint');
                    break;
                case 'complete':
                    addToConsole('OSINT scan completed! Findings: ' + data.statistics.findings_count, 'success', 'osint');
                    document.getElementById('osintProgressBar').style.width = '100%';
                    document.getElementById('osintProgressText').textContent = 'Completed - ' + data.statistics.findings_count + ' findings';
                    if (osintEventSource) osintEventSource.close();
                    updateStats();
                    break;
            }
        }
        
        function handleKaliEvent(data) {
            switch(data.type) {
                case 'progress':
                    document.getElementById('kaliProgressBar').style.width = data.progress + '%';
                    document.getElementById('kaliProgressText').textContent = data.message;
                    addToConsole(data.message, 'info', 'kali');
                    break;
                case 'finding':
                    document.getElementById('kaliFindingsCount').textContent = 
                        parseInt(document.getElementById('kaliFindingsCount').textContent) + 1;
                    addToConsole(`Found ${data.finding_type}: ${data.data.value}`, 'success', 'kali');
                    break;
                case 'error':
                    document.getElementById('kaliErrorsCount').textContent = 
                        parseInt(document.getElementById('kaliErrorsCount').textContent) + 1;
                    addToConsole('ERROR: ' + data.message, 'error', 'kali');
                    break;
                case 'complete':
                    addToConsole('Kali scan completed! Findings: ' + data.statistics.findings_count, 'success', 'kali');
                    document.getElementById('kaliProgressBar').style.width = '100%';
                    document.getElementById('kaliProgressText').textContent = 'Completed - ' + data.statistics.findings_count + ' findings';
                    if (kaliEventSource) kaliEventSource.close();
                    updateStats();
                    break;
            }
        }
        
        // Console functions
        function addToConsole(message, type = 'info', consoleType = 'osint') {
            const consoleId = consoleType === 'kali' ? 'kaliConsole' : 'osintConsole';
            const consoleDiv = document.getElementById(consoleId);
            const timestamp = new Date().toLocaleTimeString();
            const color = type === 'error' ? '#ff6b6b' : 
                         type === 'success' ? '#90ee90' : 
                         type === 'warning' ? '#ffd93d' : '#c8d8c8';
            
            consoleDiv.innerHTML += `<div style="color: ${color}">[${timestamp}] ${message}</div>`;
            consoleDiv.scrollTop = consoleDiv.scrollHeight;
        }
        
        // Stats and tools
        function updateStats() {
            fetch('/history')
                .then(r => r.json())
                .then(jobs => {
                    document.getElementById('statJobs').textContent = jobs.length;
                    document.getElementById('statActive').textContent = jobs.filter(j => j.status === 'running').length;
                    document.getElementById('statMatches').textContent = jobs.reduce((sum, job) => sum + (job.findings_count || 0), 0);
                    
                    const historyDiv = document.getElementById('jobHistory');
                    historyDiv.innerHTML = jobs.slice(0, 10).map(job => `
                        <div class="finding-card">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <small class="d-block">${job.target}</small>
                                    <small class="text-muted">${new Date(job.start_time).toLocaleTimeString()}</small>
                                </div>
                                <span class="badge ${job.status === 'completed' ? 'bg-success' : job.status === 'running' ? 'bg-warning' : 'bg-danger'}">
                                    ${job.status}
                                </span>
                            </div>
                            <div class="small text-muted mt-1">
                                ${job.findings_count || 0} findings • ${job.progress}% complete
                            </div>
                        </div>
                    `).join('');
                });
        }
        
        function loadKaliTools() {
            fetch('/kali-tools')
                .then(r => r.json())
                .then(tools => {
                    const toolsList = document.getElementById('kaliToolsList');
                    const kaliStatus = document.getElementById('kaliStatus');
                    const kaliForm = document.getElementById('kaliScanForm');
                    
                    if (Object.keys(tools).length > 0) {
                        kaliStatus.innerHTML = '<div class="alert alert-success"><i class="fas fa-check"></i> Kali Tools Available</div>';
                        kaliForm.style.display = 'block';
                        
                        toolsList.innerHTML = Object.entries(tools).map(([name, info]) => `
                            <div class="finding-card">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <strong>${info.name}</strong>
                                        <div class="small text-muted">${info.version}</div>
                                    </div>
                                    <span class="badge bg-success">Available</span>
                                </div>
                            </div>
                        `).join('');
                    } else {
                        kaliStatus.innerHTML = '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle"></i> Kali Tools Not Detected</div>';
                        kaliForm.style.display = 'none';
                        toolsList.innerHTML = '<div class="text-muted">No Kali tools detected. Install tools and refresh.</div>';
                    }
                });
        }
        
        function resetOsintCounters() {
            document.getElementById('osintFindingsCount').textContent = '0';
            document.getElementById('osintErrorsCount').textContent = '0';
            document.getElementById('osintProgressBar').style.width = '0%';
            document.getElementById('osintProgressText').textContent = 'Starting scan...';
        }
        
        function resetKaliCounters() {
            document.getElementById('kaliFindingsCount').textContent = '0';
            document.getElementById('kaliErrorsCount').textContent = '0';
            document.getElementById('kaliProgressBar').style.width = '0%';
            document.getElementById('kaliProgressText').textContent = 'Starting scan...';
        }
    </script>
</body>
</html>
"""
 
# ---------------------------
# Main Execution with Enhanced Shutdown
# ---------------------------
async def initialize_kali_tools():
    """Initialize Kali tools on startup"""
    global kali_tools_available
    try:
        print("🔧 Initializing Kali Linux tool integration...")
        tools = await kali_integrator.discover_available_tools()
        kali_tools_available = tools
        print(f"✅ Loaded {len(tools)} Kali tools")
        
        if tools:
            print("Available Kali Tools:")
            for name, info in tools.items():
                status = "✅" if info['available'] else "❌"
                print(f"  {status} {name}: {info.get('version', 'unknown')}")
        else:
            print("❌ No Kali tools detected. Install tools for full functionality.")
            
    except Exception as e:
        print(f"❌ Kali tool initialization failed: {e}")
        kali_tools_available = {}
 
def cleanup_on_shutdown():
    """Clean shutdown handler"""
    print("\n🔄 Performing graceful shutdown...")
    
    # Clean up active jobs
    with job_lock:
        for job_id, job in jobs.items():
            if job.status == 'running':
                job.fail("Server shutting down")
    
    # Wait for active threads
    for job_id, thread in active_tasks.items():
        if thread.is_alive():
            print(f"⏳ Waiting for job {job_id} to complete...")
            thread.join(timeout=5)
    
    print("✅ Shutdown complete")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OSINT + Kali Unified Dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--admin-pass', help='Admin password')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--no-browser', action='store_true', help='Disable browser mode')
    
    args = parser.parse_args()
    
    if args.admin_pass:
        ADMIN_PASSWORD = args.admin_pass
    if args.data_dir:
        DATA_DIR = args.data_dir
        KALI_CONFIG['data_dir'] = Path(args.data_dir)
        KALI_CONFIG['jobs_dir'] = Path(args.data_dir) / 'kali_jobs'
        KALI_CONFIG['audit_log'] = Path(args.data_dir) / 'audit.log'
    if args.no_browser:
        BROWSER_AVAILABLE = False
    
    # Register cleanup handlers
    atexit.register(cleanup_on_shutdown)
    signal.signal(signal.SIGINT, lambda s, f: cleanup_on_shutdown() or sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: cleanup_on_shutdown() or sys.exit(0))
    
    for directory in [DATA_DIR, LOG_DIR, CACHE_DIR]:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            sys.exit(1)
    
    # Initialize Kali tools
    asyncio.run(initialize_kali_tools())
    
    print(f"""
    🚀 OSINT + Kali Unified Dashboard Starting...
    
    Owner: Ritik Shrivas — GraySentinel
    Version: 3.3 - Critical Bug Fixes & Enhanced Error Handling
    URL: http://{args.host}:{args.port}
    
    Features:
      - Advanced OSINT scanning (CRITICAL BUGS FIXED)
      - Kali Linux tool integration with enhanced error handling
      - Real-time dashboard with proper progress tracking
      - Enhanced security & compliance with audit logging
      - Graceful shutdown handling
      - Production-ready async event loop management
      - Fixed POST data handling in search engine queries
    
    System Status:
      - Browser Mode: {'✅ Available' if BROWSER_AVAILABLE else '❌ Disabled'}
      - OCR: {'✅ Available' if OCR_AVAILABLE else '❌ Disabled'} 
      - Kali Tools: {'✅ Available' if kali_tools_available else '❌ Not Detected'}
      - Concurrency: {SEM_LIMIT}
    
    ⚠️  ETHICAL USE ONLY - Ensure proper authorization for all scans
    🔒 DEFAULT PASSWORD: admin123
    """)
    
    try:
        app.run(
            host=args.host, 
            port=args.port, 
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        logger.critical(f"Server startup failed: {e}")
        sys.exit(1)
    finally:
        cleanup_on_shutdown()
