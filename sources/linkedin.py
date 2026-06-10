"""
sources/linkedin.py
LinkedIn guest job search — no API key, no login.
Uses the public guest API endpoint that LinkedIn exposes
for unauthenticated job listing pages.
"""

import re
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
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_linkedin(keywords: list[str], location: str = "India") -> list[dict]:
    """
    Scrape LinkedIn's guest job search.
    Targets entry-level (f_E=1) jobs posted in last 24h (f_TPR=r86400).
    """
    jobs = []
    seen_titles = set()  # local dedup within this run

    for kw in keywords[:7]:  # cap at 7 keywords to avoid rate limiting
        try:
            q   = requests.utils.quote(kw)
            loc = requests.utils.quote(location)
            url = (
                "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
                f"?keywords={q}&location={loc}&f_E=1&f_TPR=r86400&start=0"
            )
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                log.debug(f"LinkedIn {kw}: HTTP {r.status_code}")
                time.sleep(3)
                continue

            content = r.text

            # Extract structured fields from the HTML response
            titles    = re.findall(
                r'class="base-search-card__title"[^>]*>\s*([^<]+?)\s*<', content
            )
            companies = re.findall(
                r'class="base-search-card__subtitle"[^>]*>\s*([^<]+?)\s*<', content
            )
            locations = re.findall(
                r'class="job-search-card__location"[^>]*>\s*([^<]+?)\s*<', content
            )
            links     = re.findall(
                r'href="(https://www\.linkedin\.com/jobs/view/[^"?]+)', content
            )

            for i, title in enumerate(titles):
                title = title.strip()
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                company  = companies[i].strip() if i < len(companies) else "Unknown"
                jid_raw  = f"{title}_{company}"
                jid      = f"li_{hashlib.md5(jid_raw.lower().encode()).hexdigest()[:10]}"

                jobs.append({
                    "job_id":      jid,
                    "title":       title,
                    "company":     company,
                    "location":    locations[i].strip() if i < len(locations) else location,
                    "description": "(Visit URL for full description — LinkedIn guest API)",
                    "url":         links[i] if i < len(links) else "https://www.linkedin.com/jobs/",
                    "posted":      date.today().isoformat(),
                    "source":      "LinkedIn",
                })

            time.sleep(2.5)  # LinkedIn is sensitive to rapid requests

        except Exception as e:
            log.error(f"LinkedIn error ({kw}): {e}")
            time.sleep(3)

    log.info(f"LinkedIn: {len(jobs)} jobs")
    return jobs