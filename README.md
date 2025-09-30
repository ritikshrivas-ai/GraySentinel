# 🛰️ GraySentinel — Military-Grade OSINT & Cyber Defense Toolkit  

**Author:** Ritik Shrivas  
**Mission:** Redefining Cybersecurity Intelligence through AI, Regex, and Real-Time Reconnaissance.  

---

## 🏴 Introduction  

GraySentinel is more than just a script. It’s a **next-gen Military Intelligence War Room** designed for OSINT, cyber defense, and investigative intelligence operations. Where most tools are either bloated or too narrow, GraySentinel sits at the intersection of **lightweight utility and advanced reconnaissance power**.  

Built to run seamlessly on **Termux (Android)** for mobility, and powerful enough for **Kali Linux environments**, GraySentinel transforms a regular device into an **intelligence hub**. It blends regex-based detection, Google Dorking, smart crawling, and integrations with proven offensive/defensive tools to give unparalleled insights into **phone numbers, emails, usernames, domains, and digital footprints**.  

Think of it as your **real-time cyber intelligence co-pilot** — running silently, efficiently, and accurately, without relying on external paid APIs or cloud dependencies.  

---

## 🚀 Key Features  

GraySentinel’s feature set is engineered to cover the entire OSINT and cyber defense lifecycle:  

1. **🔍 Regex-Powered Intelligence Engine**  
   - Scans and extracts sensitive patterns like phone numbers, emails, usernames, domains, and digital identifiers across the open web.  
   - Advanced regex rules tailored for **Indian cyber ecosystem** (PAN numbers, Aadhaar hints, GST IDs, IFSC codes, telecom formats).  
   - Enables both shallow and deep scans with normalization and confidence scoring.  

2. **🌐 Google Dorking + Smart Crawling**  
   - Preloaded with **custom dorking patterns** to hunt for leaks, public exposures, and hidden intel.  
   - Browser integration allows **real-time SERP scanning**.  
   - Supports modular dorks: Social Media exposure, Dark Web leaks, Corporate footprints, Credential dumps.  

3. **📊 Flask Web Dashboard (Military Intelligence UI)**  
   - A **single-page interface** for multiple tasks.  
   - Input, execution, and results delivered without reloading or CLI complexity.  
   - Styled to resemble **military command dashboards**, giving that tactical hacker aesthetic.  

4. **🛡 Kali Linux Integrations**  
   - Seamlessly integrates with existing industry-standard tools like:  
     - `theHarvester` (emails & subdomains)  
     - `Sherlock` (username hunting)  
     - `Maltego` (graphical intel analysis)  
     - `Recon-ng` (modular OSINT framework)  
   - Automates tool chaining to reduce analyst fatigue.  

5. **📱 Termux Ready**  
   - Fully optimized for Android penetration testers.  
   - Portable, lightweight, zero dependency on bulky VMs.  
   - Designed for **on-the-field cyber warriors** who don’t want to carry laptops everywhere.  

6. **⚡ Community-Driven R&D**  
   - Open-source & community-first.  
   - Designed to evolve with contributions from ethical hackers, researchers, and analysts worldwide.  
   - Elite contributors get pulled into a **private Discord/Intel War Room** for real-time collaboration.  

---

## 🧰 Real-World Use-Cases  

GraySentinel is built for professionals, but simple enough for curious learners:  

- **Owner Identification**  
  - Discover the **real person/organization** behind an email or phone number.  
  - Use regex + dorking to map their footprints across multiple platforms.  

- **Social Media Footprinting**  
  - Identify registered accounts across Facebook, Twitter, LinkedIn, Instagram, Telegram, etc.  
  - Correlate usernames, aliases, and handles to build a **360° digital profile**.  

- **Dark Web & Breach Hunting**  
  - Use search patterns to identify leaks, dumps, and exposed credentials.  
  - Extend into onion directories with Tor-enabled crawling modules (optional).  

- **Defensive Intel for SOC Teams**  
  - Detect where company assets (emails, phone numbers, usernames) are exposed online.  
  - Build preemptive defense measures before attackers weaponize the data.  

- **Forensics & Investigations**  
  - Leverage regex detection to find unique identifiers in forensic dumps.  
  - Trace back suspicious activity to potential owners and linked accounts.  

- **Research & Academic Intelligence**  
  - Students and researchers can use GraySentinel for **studying digital footprinting techniques** without needing high-end infra.  

---

## ⚔️ Philosophy  

> “We’re not selling fear. We’re building an **asymmetric cyber movement**.  
> GraySentinel turns cybersecurity into a **community-powered advantage**.”  
> — *Ritik Shrivas*  

Most cybersecurity tools are built around the idea of **fear-driven enterprise sales**. GraySentinel flips this model. It’s not about selling security as a cost center; it’s about building **community-powered intelligence** that grows stronger with every contributor.  

It represents a movement where **hackers, researchers, and professionals** collaborate to defend, disrupt, and innovate — without being chained to high-priced tools.  

---

## 🛠️ Technical Overview  

### Architecture  
- **Core Language:** Python (lightweight, cross-platform, community-supported).  
- **Regex Engine:** Custom pattern library tuned for global + Indian identifiers.  
- **OSINT Modules:** Independent & pluggable (email lookup, phone trace, username hunting).  
- **Dashboard:** Flask-powered single-page web app with a **Military UI theme**.  
- **Integrations:** Runs and chains Kali Linux tools automatically if present.  

### Workflow  
1. User inputs query (phone/email).  
2. Tool normalizes data → Regex Engine runs scans.  
3. Google Dorking & crawling modules collect intelligence.  
4. Matched results are validated, scored, and displayed on the dashboard.  
5. Results are saved in structured JSON for later analysis.  

---

## 📜 Installation Guide  

### 1. Clone the Repository  
```bash
git clone https://github.com/<your-username>/GraySentinel.git
cd GraySentinel
