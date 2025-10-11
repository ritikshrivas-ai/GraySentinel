<!-- BANNERS -->
<p align="center">
  <img src="https://img.shields.io/badge/GraySentinel-OSINT+Kali%20Dashboard-2d4a2d?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/License-Ethical%20Use%20Only-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Flask-2.x-darkgreen?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/Kali%20Linux%20Tools-integrated-important?style=for-the-badge&logo=linux">
  <img src="https://img.shields.io/badge/Playwright-optional-yellow?style=for-the-badge&logo=playwright">
  <img src="https://img.shields.io/badge/Bootstrap-5.x-blueviolet?style=for-the-badge&logo=bootstrap">
  <img src="https://img.shields.io/badge/BeautifulSoup-4.x-brightgreen?style=for-the-badge">
  <img src="https://img.shields.io/badge/AsyncIO-concurrent-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/AIOHTTP-fast-purple?style=for-the-badge">
</p>

<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Mono&weight=600&size=38&pause=1000&color=90EE90&center=true&vCenter=true&width=1100&lines=GraySentinel+OSINT+%2B+Kali+Linux+Dashboard;Unified+Recon+Platform+for+Researchers;Critical+Bug+Fixes+%7C+Advanced+Legal+Compliance+%7C+v3.3;Ethical+Use+Only+-+Security+First+-+No+Exploitation" alt="GraySentinel Banner"/>
</div>

---

# GraySentinel OSINT + Kali Unified Dashboard

**Owner:** Ritik Shrivas — GraySentinel / Instarecon MIL  
**Version:** 3.3 — Critical Bug Fixes & Enhanced Error Handling  
**Intended Use:** Educational, Ethical, and Authorized Security Research Only

---

## 📖 Table of Contents

- [Introduction](#introduction)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [Security & Compliance](#security--compliance)
- [Installation Guide](#installation-guide)
- [Configuration & Customization](#configuration--customization)
- [Usage Tutorial](#usage-tutorial)
- [How It Works (Detailed)](#how-it-works-detailed)
- [Advanced Usage & API](#advanced-usage--api)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Credits & Contact](#credits--contact)
- [License](#license--legal-notice)

---

## 🎯 Introduction

GraySentinel is an advanced, unified OSINT and offensive security dashboard designed for cybersecurity researchers, bug bounty hunters, and red teams. It combines deep web reconnaissance, machine learning-powered pattern extraction, and direct Kali Linux tool integration in a secure, real-time platform.

---

## 🛠️ Technology Stack

| Component              | Purpose                                       |
|------------------------|-----------------------------------------------|
| **Python 3.8+**        | Backend, async logic, automation              |
| **Flask 2.x**          | Web server and dashboard UI                   |
| **Bootstrap 5**        | Responsive tactical dashboard UI              |
| **FontAwesome**        | Icons and visual indicators                   |
| **Playwright**         | Browser automation (optional, for deep scraping)|
| **BeautifulSoup4**     | HTML parsing and extraction                   |
| **AIOHTTP**            | Async HTTP client for fast, concurrent requests|
| **AsyncIO/Threading**  | Job management, concurrency, event streaming  |
| **Kali Linux Tools**   | OSINT, secrets, metadata, and more            |
| **ML-Optimized Regex** | Enhanced pattern matching, context boosting   |
| **Audit Logging**      | Compliance, consent, legal traceability       |

**Integrated Kali Tools:**
- theHarvester
- sherlock
- amass
- exiftool
- tesseract-ocr
- gitleaks
- trufflehog
- ripgrep
- whois
- git

---

## ✨ Features

- Unified web OSINT + Kali Linux tool integration
- ML-optimized pattern library (emails, phones, payment IDs, secrets)
- Advanced dork generator for tactical search
- Real-time dashboard: live findings, progress, metrics, history
- Rate limiting, authentication, and session security
- Auto-redaction of Aadhaar, PAN, SSN, high-entropy secrets
- Legal consent prompt and audit logging for every scan
- Graceful shutdown and error recovery

---

## 🔒 Security & Compliance

- All scans require **explicit legal authorization**
- Sensitive output is auto-redacted (Aadhaar, PAN, SSN, secrets)
- Interactive consent prompt for high-risk domains (.gov, .mil, .edu, etc.)
- Rate limiting per IP to prevent abuse
- Job history and audit logs stored securely
- No exploitation, fuzzing, or unauthorized access allowed
- Authentication required (default password: `admin123`, change recommended)

---

## ⚡ Installation Guide

### Kali Linux

```bash
sudo apt-get update && sudo apt-get install -y \
  theharvester sherlock amass exiftool tesseract-ocr gitleaks trufflehog ripgrep \
  whois jq parallel timeout git curl wget python3-pip

pip install playwright beautifulsoup4 python-whois pytesseract pillow imagehash pdfminer.six aiohttp flask
```
Playwright browser drivers (optional, for deep scraping):
```bash
python -m playwright install
```

### Termux (Android)

```bash
pkg install python rust golang git curl jq
pip install playwright beautifulsoup4 python-whois pytesseract pillow aiohttp flask
```

---

## ⚙️ Configuration & Customization

- **Admin Password:**  
  `--admin-pass "yourpassword"` or set `ADMIN_PASSWORD` env var.
- **Data Directory:**  
  `--data-dir "/opt/osint_data"`
- **Disable Browser Automation:**  
  `--no-browser` flag.
- **Session Security:**  
  Set `SECRET_KEY` env var for Flask session.

Example:
```bash
python osint_kali_dashboard.py --port 5000 --host 0.0.0.0 --admin-pass "changeme" --data-dir "/opt/osint_data" --no-browser
```

---

## 🏁 Usage Tutorial

### 1. Start the Dashboard

```bash
python osint_kali_dashboard.py --port 5000 --host 127.0.0.1
```

### 2. Web Access & Authentication

- Visit: `http://127.0.0.1:5000`
- Login: `admin123` (change ASAP)

### 3. OSINT Scan

- Input Email/Phone (e.g., `user@example.com`, `9876543210`)
- Select scan intensity (Quick/Deep)
- Choose dork categories (Basic, Marketplace, Social, etc.)
- Enable browser mode (optional, for deep scraping)
- Allow sensitive data (optional, entropy-based secrets)
- **Confirm legal authorization (checkbox required)**
- Start scan — view live findings, errors, warnings, and progress.

### 4. Kali Tools Integration

- Supported features:
  - Platform Probe (Sherlock)
  - GitHub Secrets (Gitleaks)
  - EXIF Extraction (ExifTool)
  - WHOIS Lookup
  - Full OSINT Scan
- Enter target (domain, repo URL, file path)
- Set timeout
- Consent required
- Results streamed live, parsed, and redacted

### 5. Results & Job History

- View job history: status, findings, timestamps
- Download results (JSON)
- System stats: active jobs, total jobs, matches

### 6. Metrics & System Status

- `/metrics` endpoint — performance stats, active jobs, browser/Kali/OCR status

---

## 🧑‍💻 How It Works (Detailed)

### Web OSINT Engine

- **Dork Generation:** ML-optimized templates for Google, DuckDuckGo, etc.
- **Search Execution:** POST requests, random User-Agents, delays, retries.
- **Scraping:** HTML parsing, title extraction, meta tags, context snippets.
- **Pattern Matching:** Weighted regex, context boosting, entropy scoring.

### Kali Linux Tool Integration

- **Tool Discovery:** Verifies tool presence/version, caches capabilities.
- **Async Execution:** Subprocess with timeout, output parsing.
- **Findings Extraction:** Normalizes output, deduplicates, redacts sensitive info.

### ML-Optimized Pattern Library

- **Categories:** Emails, phones, payment IDs, obfuscated formats
- **Normalization:** Unicode cleanup, obfuscation de-referencing
- **Entropy Filtering:** Highlights and redacts high-entropy secrets
- **Extensible:** Add new patterns via class

### Audit Logging & Consent

- **Consent Prompt:** Interactive for sensitive domains
- **Audit Log:** Consent hash, timestamp, user, override status
- **Redaction:** Aadhaar, PAN, SSN, high-entropy secrets

### Dashboard UI

- **Responsive Design:** Bootstrap, tactical theme
- **Live Console:** Event stream updates, findings, errors
- **Tabs:** OSINT Scanner, Kali Tools, Results/History, Metrics
- **API Endpoints:** `/scan`, `/kali-scan`, `/results/<id>`, `/metrics`, `/history`, `/health`

---

## ⚡ Advanced Usage & API

- **RESTful Endpoints:**
  - `/scan` (POST): Trigger OSINT job
  - `/kali-scan` (POST): Run Kali tool
  - `/results/<job_id>`: Get job results
  - `/history`: Job list
  - `/metrics`: System stats
  - `/stream/<job_id>`: SSE event stream

- **Custom Tool Integration:**  
  Extend `KaliIntegrator` class, add new tool configs.

- **Pattern Extension:**  
  Add new regexes to `MLEnhancedPatterns`.

---

## 🩺 Troubleshooting & FAQ

- **Kali tools not detected:**  
  Ensure tools are installed and in `$PATH`. Use `/kali-tools` to verify.
- **Playwright/browser errors:**  
  Install via `pip install playwright` and run `playwright install`.
- **OCR not available:**  
  Install `tesseract-ocr` and required Python modules.
- **Permission denied:**  
  Ensure you run as a user with execution rights for required tools.
- **Jobs not completing:**  
  Check logs in `logs/` directory for errors. Increase timeout if needed.

---

## 👨‍💻 Credits & Contact

Developed by **Ritik Shrivas** (GraySentinel / Instarecon MIL)  
Contact: [ritikshrivas.ai@gmail.com](mailto:ritikshrivas.ai@gmail.com)  
For collaborations, security consulting, or custom modules.

---

## 📜 License & Legal Notice

<div align="center">
  <img src="https://img.shields.io/badge/Ethical%20Use%20Only-SECURITY%20FIRST-red?style=for-the-badge">
</div>

> **GraySentinel OSINT + Kali Dashboard is for educational, research, and authorized security use only.**
> 
> - You must have explicit legal authorization for all targets scanned.
> - All usage is logged and auditable.
> - Do **not** use for illegal, unethical, or unauthorized activities.
> - By using this tool, you agree to comply with all applicable laws and regulations.

---

_For bug reports, feature requests, or contributions, open an issue or PR._
