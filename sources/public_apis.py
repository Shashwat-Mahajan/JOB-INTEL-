"""
sources/public_apis.py
Free public job APIs — no authentication required.
"""

import time
import hashlib
import logging
import requests
from datetime import datetime, date

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _make_id(source: str, val: str) -> str:
    h = hashlib.md5(f"{source}_{val}".lower().encode()).hexdigest()[:10]
    return f"{source}_{h}"


def _is_recent(date_str: str, days: int = 3) -> bool:
    """Return True if the posting date is within the last N days."""
    if not date_str:
        return True  # assume recent if no date
    try:
        pub = datetime.fromisoformat(date_str[:10])
        return (datetime.now() - pub).days <= days
    except Exception:
        return True


def fetch_remotive(keywords: list[str]) -> list[dict]:
    """Remotive — free remote tech jobs API, no auth needed."""
    jobs = []
    try:
        for kw in keywords:
            url = f"https://remotive.com/api/remote-jobs?search={requests.utils.quote(kw)}&limit=20"
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            for j in r.json().get("jobs", []):
                if not _is_recent(j.get("publication_date", ""), days=3):
                    continue
                jobs.append({
                    "job_id":      _make_id("remotive", str(j.get("id", ""))),
                    "title":       j.get("title", ""),
                    "company":     j.get("company_name", ""),
                    "location":    j.get("candidate_required_location", "Remote"),
                    "description": (j.get("description") or "")[:1200],
                    "url":         j.get("url", ""),
                    "posted":      (j.get("publication_date") or "")[:10],
                    "source":      "Remotive",
                })
            time.sleep(0.8)
    except Exception as e:
        log.error(f"Remotive error: {e}")
    log.info(f"Remotive: {len(jobs)} jobs")
    return jobs


def fetch_arbeitnow(keywords: list[str]) -> list[dict]:
    """Arbeitnow — free tech job board API, no auth needed."""
    jobs = []
    try:
        r = requests.get(
            "https://www.arbeitnow.com/api/job-board-api",
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code == 200:
            for j in r.json().get("data", []):
                combined = (j.get("title", "") + " " + j.get("description", "")).lower()
                if not any(kw.lower() in combined for kw in keywords):
                    continue
                jobs.append({
                    "job_id":      _make_id("arbeitnow", j.get("slug", j.get("title", ""))),
                    "title":       j.get("title", ""),
                    "company":     j.get("company_name", ""),
                    "location":    j.get("location", "Remote"),
                    "description": (j.get("description") or "")[:1200],
                    "url":         j.get("url", ""),
                    "posted":      date.today().isoformat(),
                    "source":      "Arbeitnow",
                })
    except Exception as e:
        log.error(f"Arbeitnow error: {e}")
    log.info(f"Arbeitnow: {len(jobs)} jobs")
    return jobs


def fetch_jobicy(keywords: list[str]) -> list[dict]:
    """Jobicy — free remote job board API, tag-based search."""
    jobs = []
    tags = ["software-engineer", "machine-learning", "data-engineer", "backend", "python"]
    try:
        for tag in tags:
            r = requests.get(
                f"https://jobicy.com/api/v2/remote-jobs?tag={tag}&count=20",
                headers=HEADERS,
                timeout=15,
            )
            if r.status_code != 200:
                continue
            for j in r.json().get("jobs", []):
                combined = (j.get("jobTitle", "") + " " + j.get("jobDescription", "")).lower()
                if not any(kw.lower() in combined for kw in keywords):
                    continue
                jobs.append({
                    "job_id":      _make_id("jobicy", str(j.get("id", j.get("jobTitle", "")))),
                    "title":       j.get("jobTitle", ""),
                    "company":     j.get("companyName", ""),
                    "location":    j.get("jobGeo", "Remote"),
                    "description": (j.get("jobDescription") or "")[:1200],
                    "url":         j.get("url", ""),
                    "posted":      (j.get("pubDate") or "")[:10],
                    "source":      "Jobicy",
                })
            time.sleep(0.5)
    except Exception as e:
        log.error(f"Jobicy error: {e}")
    log.info(f"Jobicy: {len(jobs)} jobs")
    return jobs