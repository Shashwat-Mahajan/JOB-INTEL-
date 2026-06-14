"""
sources/linkedin.py
LinkedIn guest job search — no API key, no login required.
Searches multiple keyword+location combos across 2 pages each.
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
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# All keyword+location combos to search
# f_E=1,2 = entry level + associate
# f_TPR=r604800 = posted in last 7 days
LINKEDIN_SEARCHES = [
    ("generative AI engineer",            "India"),
    ("LLM engineer",                      "India"),
    ("machine learning engineer fresher", "India"),
    ("AI engineer 2027",                  "India"),
    ("software engineer AI",              "India"),
    ("backend engineer python",           "India"),
    ("SDE fresher",                       "India"),
    ("data scientist fresher",            "India"),
    ("NLP engineer",                      "India"),
    ("MLOps engineer",                    "India"),
    ("software engineer",                 "Bengaluru, Karnataka, India"),
    ("machine learning",                  "Bengaluru, Karnataka, India"),
    ("AI engineer",                       "Hyderabad, Telangana, India"),
    ("software engineer",                 "Pune, Maharashtra, India"),
    ("data scientist",                    "Hyderabad, Telangana, India"),
    ("software engineer",                 "Mumbai, Maharashtra, India"),
]


def fetch_linkedin(keywords: list, location: str = "India") -> list:
    jobs        = []
    seen_dedup  = set()

    # Combine passed keywords with our expanded list
    all_searches = [(kw, location) for kw in keywords[:4]]
    all_searches += LINKEDIN_SEARCHES

    for kw, loc in all_searches:
        for start in [0, 25]:  # 2 pages per keyword
            try:
                q   = requests.utils.quote(kw)
                l   = requests.utils.quote(loc)
                url = (
                    "https://www.linkedin.com/jobs-guest/jobs/api/"
                    "seeMoreJobPostings/search"
                    f"?keywords={q}&location={l}"
                    f"&f_E=1,2"
                    f"&f_TPR=r604800"
                    f"&start={start}"
                )
                r = requests.get(url, headers=HEADERS, timeout=20)
                if r.status_code != 200:
                    time.sleep(3)
                    continue

                content   = r.text
                titles    = re.findall(
                    r'class="base-search-card__title"[^>]*>\s*([^<]+?)\s*<',
                    content
                )
                companies = re.findall(
                    r'class="base-search-card__subtitle"[^>]*>\s*([^<]+?)\s*<',
                    content
                )
                locations = re.findall(
                    r'class="job-search-card__location"[^>]*>\s*([^<]+?)\s*<',
                    content
                )
                links     = re.findall(
                    r'href="(https://www\.linkedin\.com/jobs/view/[^"?]+)',
                    content
                )

                for i, title in enumerate(titles):
                    title     = title.strip()
                    company   = companies[i].strip() if i < len(companies) else "Unknown"
                    dedup_key = f"{title}_{company}".lower()

                    if dedup_key in seen_dedup:
                        continue
                    seen_dedup.add(dedup_key)

                    jid = f"li_{hashlib.md5(dedup_key.encode()).hexdigest()[:10]}"
                    jobs.append({
                        "job_id":      jid,
                        "title":       title,
                        "company":     company,
                        "location":    locations[i].strip() if i < len(locations) else loc,
                        "description": "(Visit URL for full job description)",
                        "url":         links[i] if i < len(links) else "https://www.linkedin.com/jobs/",
                        "posted":      date.today().isoformat(),
                        "source":      "LinkedIn",
                    })

                time.sleep(3)  # LinkedIn needs a polite delay between requests

            except Exception as e:
                log.error(f"LinkedIn error ({kw}): {e}")
                time.sleep(4)

    log.info(f"LinkedIn: {len(jobs)} jobs")
    return jobs