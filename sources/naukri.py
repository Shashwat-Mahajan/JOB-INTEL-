"""
sources/naukri.py
Naukri fresher job scraper.
Extracts structured data from JSON-LD blocks embedded in Naukri pages.
No API key required — uses public-facing search URLs.
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
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Naukri fresher search pages — these target 0–2 year experience
NAUKRI_SLUGS = [
    "generative-ai-jobs",
    "machine-learning-engineer-fresher-jobs",
    "software-developer-fresher-jobs-2026",
    "ai-engineer-jobs",
    "nlp-engineer-fresher-jobs",
    "python-developer-fresher-jobs",
    "data-scientist-fresher-jobs",
    "backend-developer-fresher-jobs",
]


def fetch_naukri() -> list[dict]:
    """
    Scrape Naukri's structured job data from JSON-LD blocks.
    These are embedded in the page source and follow the JobPosting schema.
    """
    jobs = []
    seen_ids = set()

    for slug in NAUKRI_SLUGS:
        try:
            url = f"https://www.naukri.com/{slug}"
            r   = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                log.debug(f"Naukri {slug}: HTTP {r.status_code}")
                time.sleep(1)
                continue

            # Extract all JSON-LD blocks from the page
            ld_blocks = re.findall(
                r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
                r.text,
                re.DOTALL,
            )

            for block in ld_blocks:
                try:
                    data = json.loads(block.strip())
                except json.JSONDecodeError:
                    continue

                # Handle both single items and @graph arrays
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

                    # Build a stable ID from title + company
                    company = job.get("hiringOrganization", {}).get("name", "Unknown")
                    raw_id  = f"naukri_{title}_{company}".lower()
                    jid     = "nk_" + hashlib.md5(raw_id.encode()).hexdigest()[:10]

                    if jid in seen_ids:
                        continue
                    seen_ids.add(jid)

                    # Extract location
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