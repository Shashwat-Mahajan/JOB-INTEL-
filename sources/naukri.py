"""
sources/naukri.py
Naukri fresher job scraper — extracts JSON-LD structured data.
No API key required.
"""

import re
import json
import hashlib
import logging
import requests
import time
from datetime import date

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

NAUKRI_SLUGS = [
    # Full-time fresher jobs
    "generative-ai-jobs",
    "machine-learning-engineer-fresher-jobs",
    "software-developer-fresher-jobs-2026",
    "ai-engineer-jobs",
    "nlp-engineer-fresher-jobs",
    "python-developer-fresher-jobs",
    "data-scientist-fresher-jobs",
    "backend-developer-fresher-jobs",
    "software-engineer-jobs-for-freshers",
    "artificial-intelligence-jobs",
    "deep-learning-jobs",
    # Internships
    "machine-learning-internship-jobs",
    "artificial-intelligence-internship-jobs",
    "software-developer-internship-jobs",
    "data-science-internship-jobs",
    "python-internship-jobs",
    "deep-learning-internship-jobs",
    "nlp-internship-jobs",
    "generative-ai-internship-jobs",
]


def fetch_naukri() -> list:
    """Scrape Naukri's structured job data from JSON-LD blocks."""
    jobs     = []
    seen_ids = set()

    for slug in NAUKRI_SLUGS:
        try:
            url = f"https://www.naukri.com/{slug}"
            r   = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                time.sleep(1)
                continue

            ld_blocks = re.findall(
                r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>'
                r'(.*?)</script>',
                r.text, re.DOTALL
            )

            for block in ld_blocks:
                try:
                    data = json.loads(block.strip())
                except json.JSONDecodeError:
                    continue

                items = []
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    if data.get("@type") == "JobPosting":
                        items = [data]
                    elif "@graph" in data:
                        items = data["@graph"]
                    else:
                        items = data.get("itemListElement", [data])

                for item in items:
                    job = item.get("item", item)
                    if job.get("@type") != "JobPosting":
                        continue

                    title = job.get("title", "")
                    if not title:
                        continue

                    company = job.get("hiringOrganization", {}).get("name", "Unknown")
                    raw_id  = f"{title}_{company}".lower()
                    jid     = "nk_" + hashlib.md5(raw_id.encode()).hexdigest()[:10]

                    if jid in seen_ids:
                        continue
                    seen_ids.add(jid)

                    loc_obj  = job.get("jobLocation", {})
                    address  = loc_obj.get("address", {}) if isinstance(loc_obj, dict) else {}
                    location = address.get("addressLocality", "India")

                    jobs.append({
                        "job_id":      jid,
                        "title":       title,
                        "company":     company,
                        "location":    str(location),
                        "description": (job.get("description") or "")[:1200],
                        "url":         job.get("url", url),
                        "posted":      (job.get("datePosted") or "")[:10] or date.today().isoformat(),
                        "source":      "Naukri",
                    })

            time.sleep(1.5)

        except Exception as e:
            log.error(f"Naukri error ({slug}): {e}")
            time.sleep(2)

    log.info(f"Naukri: {len(jobs)} jobs")
    return jobs