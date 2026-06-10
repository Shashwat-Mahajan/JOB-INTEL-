"""
sources/career_portals.py
Scrapes 13 company career portals directly.
Uses public JSON APIs — Greenhouse, Lever, Workday, proprietary.
No login or API keys required for any of these.
"""

import json
import time
import hashlib
import logging
import requests
from datetime import date

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
}

# Keywords for fast pre-filter BEFORE sending to LLM (saves Groq tokens)
AI_KEYWORDS = [
    "ai", "ml", "machine learning", "generative", "llm", "nlp",
    "deep learning", "data science", "artificial intelligence",
    "genai", "langchain", "python", "backend", "software engineer",
    "sde", "software developer", "data engineer", "platform engineer",
]


def _id(company: str, val: str) -> str:
    h = hashlib.md5(f"{company}_{val}".lower().encode()).hexdigest()[:8]
    return f"{company.lower()[:6]}_{h}"


def _is_relevant(title: str, desc: str = "") -> bool:
    """Quick pre-filter — must contain at least one AI/tech keyword."""
    combined = (title + " " + desc).lower()
    return any(kw in combined for kw in AI_KEYWORDS)


def _get(url: str, timeout: int = 20) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r
        log.debug(f"  {url[:80]}  →  HTTP {r.status_code}")
    except Exception as e:
        log.debug(f"  {url[:80]}  →  {e}")
    return None


# ── Greenhouse boards (Swiggy, Meesho, Atlassian) ────────────────────────────

def _fetch_greenhouse(company: str, board: str) -> list[dict]:
    """Generic fetcher for any company using Greenhouse ATS."""
    jobs = []
    r = _get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true")
    if not r:
        return jobs
    for j in r.json().get("jobs", []):
        title = j.get("title", "")
        if not _is_relevant(title, j.get("content", "")):
            continue
        jobs.append({
            "job_id":      _id(board, str(j.get("id", title))),
            "title":       title,
            "company":     company,
            "location":    j.get("location", {}).get("name", "India"),
            "description": (j.get("content") or "")[:1200],
            "url":         j.get("absolute_url", f"https://boards.greenhouse.io/{board}"),
            "posted":      (j.get("updated_at") or "")[:10] or date.today().isoformat(),
            "source":      f"{company} Careers",
        })
    return jobs


def fetch_swiggy() -> list[dict]:
    jobs = _fetch_greenhouse("Swiggy", "swiggy")
    log.info(f"Swiggy: {len(jobs)} jobs")
    return jobs


def fetch_meesho() -> list[dict]:
    jobs = _fetch_greenhouse("Meesho", "meesho")
    log.info(f"Meesho: {len(jobs)} jobs")
    return jobs


def fetch_atlassian() -> list[dict]:
    """Atlassian — Greenhouse board, filtered to India/remote only."""
    raw = _fetch_greenhouse("Atlassian", "atlassian")
    jobs = [
        j for j in raw
        if any(w in j["location"].lower() for w in ["india", "remote", "bangalore", "hyderabad"])
    ]
    log.info(f"Atlassian: {len(jobs)} jobs")
    return jobs


# ── Lever boards (Razorpay, PhonePe) ─────────────────────────────────────────

def _fetch_lever(company: str, slug: str) -> list[dict]:
    """Generic fetcher for any company using Lever ATS."""
    jobs = []
    r = _get(f"https://api.lever.co/v0/postings/{slug}?mode=json")
    if not r:
        return jobs
    for j in r.json():
        title = j.get("text", "")
        if not _is_relevant(title, j.get("description", "")):
            continue
        cats = j.get("categories", {})
        jobs.append({
            "job_id":      _id(slug, j.get("id", title)),
            "title":       title,
            "company":     company,
            "location":    cats.get("location", "India"),
            "description": (j.get("description") or "")[:1200],
            "url":         j.get("hostedUrl", f"https://jobs.lever.co/{slug}"),
            "posted":      date.today().isoformat(),
            "source":      f"{company} Careers",
        })
    return jobs


def fetch_razorpay() -> list[dict]:
    jobs = _fetch_lever("Razorpay", "razorpay")
    log.info(f"Razorpay: {len(jobs)} jobs")
    return jobs


def fetch_phonepe() -> list[dict]:
    jobs = _fetch_lever("PhonePe", "phonepe")
    log.info(f"PhonePe: {len(jobs)} jobs")
    return jobs


# ── Amazon ────────────────────────────────────────────────────────────────────

def fetch_amazon() -> list[dict]:
    jobs = []
    try:
        for q in ["Machine Learning", "Software Development Engineer", "AI Engineer", "Data Scientist"]:
            url = (
                "https://www.amazon.jobs/en/search.json"
                f"?base_query={requests.utils.quote(q)}"
                "&loc_query=India&job_type=Full-Time"
                "&experience_ids=ENTRY_LEVEL&result_limit=20"
            )
            r = _get(url)
            if not r:
                continue
            for j in r.json().get("jobs", []):
                title = j.get("title", "")
                if not _is_relevant(title, j.get("description_short", "")):
                    continue
                jobs.append({
                    "job_id":      _id("amazon", title + str(j.get("id_icims", ""))),
                    "title":       title,
                    "company":     "Amazon",
                    "location":    j.get("location", "India"),
                    "description": (j.get("description_short") or "")[:1200],
                    "url":         "https://www.amazon.jobs" + j.get("job_path", ""),
                    "posted":      (j.get("posted_date") or "")[:10],
                    "source":      "Amazon Careers",
                })
            time.sleep(1)
    except Exception as e:
        log.error(f"Amazon error: {e}")
    log.info(f"Amazon: {len(jobs)} jobs")
    return jobs


# ── Google ────────────────────────────────────────────────────────────────────

def fetch_google() -> list[dict]:
    jobs = []
    try:
        for q in ["Machine Learning Engineer", "Software Engineer AI", "Software Engineer"]:
            url = (
                f"https://careers.google.com/api/v3/search/"
                f"?q={requests.utils.quote(q)}"
                "&location=India&jex=ENTRY_LEVEL&page_size=20&page=1"
            )
            r = _get(url)
            if not r:
                continue
            for j in r.json().get("jobs", []):
                title = j.get("title", "")
                desc  = " ".join(
                    j.get("description", {}).get("responsibilities", [])
                    + j.get("description", {}).get("qualifications", [])
                )
                if not _is_relevant(title, desc):
                    continue
                locs = j.get("locations", [{}])
                jobs.append({
                    "job_id":      _id("google", j.get("id", title)),
                    "title":       title,
                    "company":     "Google",
                    "location":    locs[0].get("display", "India") if locs else "India",
                    "description": desc[:1200],
                    "url":         f"https://careers.google.com/jobs/results/{j.get('id','')}",
                    "posted":      (j.get("publish_date") or "")[:10],
                    "source":      "Google Careers",
                })
            time.sleep(1.2)
    except Exception as e:
        log.error(f"Google error: {e}")
    log.info(f"Google: {len(jobs)} jobs")
    return jobs


# ── Microsoft ─────────────────────────────────────────────────────────────────

def fetch_microsoft() -> list[dict]:
    jobs = []
    try:
        for q in ["AI Engineer", "Software Engineer", "Machine Learning"]:
            url = (
                "https://jobs.careers.microsoft.com/global/en/search"
                f"?q={requests.utils.quote(q)}"
                "&lc=India&exp=Students+and+recent+graduates&pgSz=20&pg=1&format=json"
            )
            r = _get(url)
            if not r:
                continue
            for j in r.json().get("operationResult", {}).get("result", {}).get("jobs", []):
                title = j.get("title", "")
                desc  = j.get("descriptionTeaser", "")
                if not _is_relevant(title, desc):
                    continue
                jobs.append({
                    "job_id":      _id("microsoft", str(j.get("jobId", title))),
                    "title":       title,
                    "company":     "Microsoft",
                    "location":    j.get("location", "India"),
                    "description": desc[:1200],
                    "url":         f"https://jobs.careers.microsoft.com/global/en/job/{j.get('jobId','')}",
                    "posted":      (j.get("postingDate") or "")[:10],
                    "source":      "Microsoft Careers",
                })
            time.sleep(1.2)
    except Exception as e:
        log.error(f"Microsoft error: {e}")
    log.info(f"Microsoft: {len(jobs)} jobs")
    return jobs


# ── Flipkart ──────────────────────────────────────────────────────────────────

def fetch_flipkart() -> list[dict]:
    jobs = []
    try:
        for q in ["machine learning", "software engineer", "data science", "ai"]:
            url = f"https://www.flipkartcareers.com/api/jobs?keyword={requests.utils.quote(q)}"
            r = _get(url)
            if not r:
                continue
            data = r.json()
            items = data if isinstance(data, list) else data.get("jobs", [])
            for j in items:
                title = j.get("jobTitle", j.get("title", ""))
                if not _is_relevant(title, j.get("jobDescription", "")):
                    continue
                jobs.append({
                    "job_id":      _id("flipkart", str(j.get("jobId", title))),
                    "title":       title,
                    "company":     "Flipkart",
                    "location":    j.get("location", "Bengaluru"),
                    "description": (j.get("jobDescription") or "")[:1200],
                    "url":         f"https://www.flipkartcareers.com/#!/jobdetail/{j.get('jobId','')}",
                    "posted":      (j.get("postedDate") or "")[:10] or date.today().isoformat(),
                    "source":      "Flipkart Careers",
                })
            time.sleep(0.8)
    except Exception as e:
        log.error(f"Flipkart error: {e}")
    log.info(f"Flipkart: {len(jobs)} jobs")
    return jobs


# ── Walmart Global Tech ───────────────────────────────────────────────────────

def fetch_walmart() -> list[dict]:
    jobs = []
    try:
        for q in ["Machine Learning", "Software Engineer", "AI", "Data Scientist"]:
            url = (
                f"https://careers.walmart.com/api/jobs"
                f"?q={requests.utils.quote(q)}&location=India"
                "&careerLevel=entry+level&page=1&pageSize=20"
            )
            r = _get(url)
            if not r:
                continue
            data = r.json()
            items = data if isinstance(data, list) else data.get("jobs", [])
            for j in items:
                title = j.get("title", j.get("jobTitle", ""))
                desc  = j.get("shortDescription", j.get("description", ""))
                if not _is_relevant(title, desc):
                    continue
                jobs.append({
                    "job_id":      _id("walmart", str(j.get("jobId", title))),
                    "title":       title,
                    "company":     "Walmart Global Tech",
                    "location":    j.get("location", "Bengaluru"),
                    "description": desc[:1200],
                    "url":         f"https://careers.walmart.com/us/jobs/{j.get('jobId','')}",
                    "posted":      (j.get("postedDate") or "")[:10] or date.today().isoformat(),
                    "source":      "Walmart Careers",
                })
            time.sleep(1)
    except Exception as e:
        log.error(f"Walmart error: {e}")
    log.info(f"Walmart: {len(jobs)} jobs")
    return jobs


# ── Adobe ─────────────────────────────────────────────────────────────────────

def fetch_adobe() -> list[dict]:
    jobs = []
    try:
        for q in ["Machine Learning", "AI Engineer", "Software Engineer"]:
            url = (
                f"https://adobe.wd5.myworkdayjobs.com/wday/cxs/adobe/ADC/jobs"
                f"?q={requests.utils.quote(q)}&locations=India"
            )
            r = _get(url)
            if not r:
                continue
            for j in r.json().get("jobPostings", []):
                title = j.get("title", "")
                if not _is_relevant(title, j.get("locationsText", "")):
                    continue
                ext = j.get("externalPath", "").split("/")[-1]
                jobs.append({
                    "job_id":      _id("adobe", ext or title),
                    "title":       title,
                    "company":     "Adobe",
                    "location":    j.get("locationsText", "India"),
                    "description": (j.get("jobDescription") or "")[:1200],
                    "url":         f"https://adobe.wd5.myworkdayjobs.com/ADC{j.get('externalPath','')}",
                    "posted":      (j.get("postedOn") or "")[:10] or date.today().isoformat(),
                    "source":      "Adobe Careers",
                })
            time.sleep(1)
    except Exception as e:
        log.error(f"Adobe error: {e}")
    log.info(f"Adobe: {len(jobs)} jobs")
    return jobs


# ── Instahyre ─────────────────────────────────────────────────────────────────

def fetch_instahyre() -> list[dict]:
    jobs = []
    try:
        for kw in ["machine learning", "generative ai", "software engineer", "ai engineer"]:
            url = f"https://www.instahyre.com/api/v1/job/?q={requests.utils.quote(kw)}&limit=30"
            r = _get(url)
            if not r:
                continue
            data = r.json()
            items = data if isinstance(data, list) else data.get("results", data.get("jobs", []))
            for j in items:
                title = j.get("designation", j.get("title", ""))
                if not _is_relevant(title, j.get("description", "")):
                    continue
                company_obj = j.get("company", {})
                company = (
                    company_obj.get("name", "") if isinstance(company_obj, dict)
                    else j.get("company_name", "Unknown")
                )
                jobs.append({
                    "job_id":      _id("instahyre", str(j.get("id", title))),
                    "title":       title,
                    "company":     company,
                    "location":    j.get("location", "India"),
                    "description": (j.get("description") or "")[:1200],
                    "url":         f"https://www.instahyre.com/job-{j.get('id','')}",
                    "posted":      (j.get("created_at") or "")[:10] or date.today().isoformat(),
                    "source":      "Instahyre",
                })
            time.sleep(1)
    except Exception as e:
        log.error(f"Instahyre error: {e}")
    log.info(f"Instahyre: {len(jobs)} jobs")
    return jobs


# ── Master fetcher ────────────────────────────────────────────────────────────

def fetch_all_career_portals() -> list[dict]:
    """Run all portal fetchers and return combined deduplicated list."""
    all_jobs = []
    fetchers = [
        ("Swiggy",        fetch_swiggy),
        ("Meesho",        fetch_meesho),
        ("Atlassian",     fetch_atlassian),
        ("Razorpay",      fetch_razorpay),
        ("PhonePe",       fetch_phonepe),
        ("Amazon",        fetch_amazon),
        ("Google",        fetch_google),
        ("Microsoft",     fetch_microsoft),
        ("Flipkart",      fetch_flipkart),
        ("Walmart",       fetch_walmart),
        ("Adobe",         fetch_adobe),
        ("Instahyre",     fetch_instahyre),
    ]
    for name, fn in fetchers:
        try:
            result = fn()
            all_jobs.extend(result)
        except Exception as e:
            log.error(f"Career portal error — {name}: {e}")
        time.sleep(1)  # polite delay between companies

    log.info(f"Career portals total: {len(all_jobs)} jobs")
    return all_jobs