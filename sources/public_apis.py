"""
sources/public_apis.py — expanded with more reliable sources
"""

import time
import hashlib
import logging
import requests
from datetime import datetime, date

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _make_id(source: str, val: str) -> str:
    h = hashlib.md5(f"{source}_{val}".lower().encode()).hexdigest()[:10]
    return f"{source}_{h}"


def _is_recent(date_str: str, days: int = 7) -> bool:
    if not date_str:
        return True
    try:
        pub = datetime.fromisoformat(date_str[:10])
        return (datetime.now() - pub).days <= days
    except Exception:
        return True


def fetch_remotive(keywords: list) -> list:
    """Remotive public API — remote tech jobs, no auth."""
    jobs = []
    # Use broader categories instead of specific keywords
    categories = [
        "software-dev",
        "data",
        "devops-sysadmin",
        "product",
    ]
    seen_ids = set()
    try:
        # First try category-based fetch (returns more results)
        for cat in categories:
            url = f"https://remotive.com/api/remote-jobs?category={cat}&limit=50"
            r   = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            for j in r.json().get("jobs", []):
                jid = str(j.get("id", ""))
                if jid in seen_ids:
                    continue
                seen_ids.add(jid)
                title = j.get("title", "").lower()
                # Filter for relevant roles
                if not any(k in title for k in [
                    "engineer", "developer", "scientist", "analyst",
                    "ml", "ai", "data", "backend", "python", "nlp"
                ]):
                    continue
                jobs.append({
                    "job_id":      _make_id("remotive", jid),
                    "title":       j.get("title", ""),
                    "company":     j.get("company_name", ""),
                    "location":    j.get("candidate_required_location", "Worldwide"),
                    "description": (j.get("description") or "")[:1200],
                    "url":         j.get("url", ""),
                    "posted":      (j.get("publication_date") or "")[:10],
                    "source":      "Remotive",
                })
            time.sleep(0.5)
        log.info(f"Remotive: {len(jobs)} jobs")
    except Exception as e:
        log.error(f"Remotive error: {e}")
    return jobs


def fetch_arbeitnow(keywords: list) -> list:
    """Arbeitnow — European + remote tech jobs."""
    jobs = []
    try:
        r = requests.get(
            "https://www.arbeitnow.com/api/job-board-api",
            headers=HEADERS, timeout=15
        )
        if r.status_code == 200:
            for j in r.json().get("data", []):
                combined = (j.get("title","") + " " + j.get("description","")).lower()
                if not any(k.lower() in combined for k in keywords):
                    continue
                jobs.append({
                    "job_id":      _make_id("arbeitnow", j.get("slug", j.get("title",""))),
                    "title":       j.get("title", ""),
                    "company":     j.get("company_name", ""),
                    "location":    j.get("location", "Remote"),
                    "description": (j.get("description") or "")[:1200],
                    "url":         j.get("url", ""),
                    "posted":      date.today().isoformat(),
                    "source":      "Arbeitnow",
                })
        log.info(f"Arbeitnow: {len(jobs)} jobs")
    except Exception as e:
        log.error(f"Arbeitnow error: {e}")
    return jobs


def fetch_jobicy(keywords: list) -> list:
    """Jobicy — remote tech jobs."""
    jobs = []
    tags = ["software-engineer", "machine-learning", "data-engineer",
            "backend", "python", "ai"]
    try:
        for tag in tags:
            r = requests.get(
                f"https://jobicy.com/api/v2/remote-jobs?tag={tag}&count=20",
                headers=HEADERS, timeout=15
            )
            if r.status_code != 200:
                continue
            for j in r.json().get("jobs", []):
                combined = (j.get("jobTitle","") + " " + j.get("jobDescription","")).lower()
                if not any(k.lower() in combined for k in keywords):
                    continue
                jobs.append({
                    "job_id":      _make_id("jobicy", str(j.get("id", j.get("jobTitle","")))),
                    "title":       j.get("jobTitle", ""),
                    "company":     j.get("companyName", ""),
                    "location":    j.get("jobGeo", "Remote"),
                    "description": (j.get("jobDescription") or "")[:1200],
                    "url":         j.get("url", ""),
                    "posted":      (j.get("pubDate") or "")[:10],
                    "source":      "Jobicy",
                })
            time.sleep(0.5)
        log.info(f"Jobicy: {len(jobs)} jobs")
    except Exception as e:
        log.error(f"Jobicy error: {e}")
    return jobs


def fetch_himalayas(keywords: list) -> list:
    """Himalayas.app — free remote jobs API, no auth, great for tech."""
    jobs = []
    try:
        for kw in ["machine learning", "software engineer", "AI engineer",
                   "backend engineer", "data scientist", "python developer"]:
            url = f"https://himalayas.app/jobs/api?q={requests.utils.quote(kw)}&limit=20"
            r   = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            data  = r.json()
            items = data if isinstance(data, list) else data.get("jobs", [])
            for j in items:
                title = j.get("title", j.get("position", ""))
                jobs.append({
                    "job_id":      _make_id("himalayas", str(j.get("id", title))),
                    "title":       title,
                    "company":     j.get("company", {}).get("name", j.get("companyName", "")),
                    "location":    j.get("location", "Remote"),
                    "description": (j.get("description") or "")[:1200],
                    "url":         j.get("url", j.get("applyUrl", "")),
                    "posted":      (j.get("createdAt") or "")[:10] or date.today().isoformat(),
                    "source":      "Himalayas",
                })
            time.sleep(0.8)
        log.info(f"Himalayas: {len(jobs)} jobs")
    except Exception as e:
        log.error(f"Himalayas error: {e}")
    return jobs


def fetch_otta(keywords: list) -> list:
    """Otta — tech jobs platform, good for startups and scale-ups."""
    jobs = []
    try:
        url = "https://app.otta.com/api/jobs/search"
        for kw in ["machine learning", "software engineer", "AI"]:
            try:
                r = requests.post(
                    url,
                    json={"query": kw, "locations": ["India"], "remote": True},
                    headers={**HEADERS, "Content-Type": "application/json"},
                    timeout=15,
                )
                if r.status_code != 200:
                    continue
                for j in r.json().get("results", r.json().get("jobs", [])):
                    title = j.get("title", j.get("function", ""))
                    jobs.append({
                        "job_id":      _make_id("otta", str(j.get("id", title))),
                        "title":       title,
                        "company":     j.get("company", {}).get("name", ""),
                        "location":    j.get("location", "Remote"),
                        "description": (j.get("bullets", {}).get("engineering", [""])[0] or "")[:1200],
                        "url":         f"https://app.otta.com/jobs/{j.get('externalId', j.get('id',''))}",
                        "posted":      date.today().isoformat(),
                        "source":      "Otta",
                    })
                time.sleep(1)
            except Exception:
                pass
        log.info(f"Otta: {len(jobs)} jobs")
    except Exception as e:
        log.error(f"Otta error: {e}")
    return jobs


def fetch_freshersworld(keywords: list) -> list:
    """Freshersworld — India's dedicated fresher job portal."""
    jobs = []
    try:
        searches = [
            "https://www.freshersworld.com/jobs/jobsearch/machine-learning-engineer-jobs-for-freshers?job_type=fresher",
            "https://www.freshersworld.com/jobs/jobsearch/artificial-intelligence-jobs-for-freshers?job_type=fresher",
            "https://www.freshersworld.com/jobs/jobsearch/software-engineer-jobs-for-freshers?job_type=fresher",
            "https://www.freshersworld.com/jobs/jobsearch/python-developer-jobs-for-freshers?job_type=fresher",
            "https://www.freshersworld.com/jobs/jobsearch/data-scientist-jobs-for-freshers?job_type=fresher",
        ]
        from bs4 import BeautifulSoup
        import re

        for url in searches:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            # Extract JSON-LD job postings
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    import json as _json
                    data  = _json.loads(script.string)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if item.get("@type") != "JobPosting":
                            continue
                        title = item.get("title", "")
                        org   = item.get("hiringOrganization", {})
                        loc   = item.get("jobLocation", {})
                        addr  = loc.get("address", {}) if isinstance(loc, dict) else {}
                        jobs.append({
                            "job_id":      _make_id("freshersworld", title + item.get("url","")),
                            "title":       title,
                            "company":     org.get("name","") if isinstance(org,dict) else "",
                            "location":    addr.get("addressLocality","India"),
                            "description": (item.get("description") or "")[:1200],
                            "url":         item.get("url", url),
                            "posted":      (item.get("datePosted") or "")[:10] or date.today().isoformat(),
                            "source":      "Freshersworld",
                        })
                except Exception:
                    pass
            time.sleep(1.5)
        log.info(f"Freshersworld: {len(jobs)} jobs")
    except Exception as e:
        log.error(f"Freshersworld error: {e}")
    return jobs