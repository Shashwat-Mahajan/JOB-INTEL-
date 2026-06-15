"""
sources/career_portals.py
Updated June 2026 — fixed endpoints, better error handling.
"""

import json
import time
import hashlib
import logging
import requests
from datetime import date
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}

AI_KEYWORDS = [
    "ai", "ml", "machine learning", "generative", "llm", "nlp",
    "deep learning", "data science", "artificial intelligence",
    "genai", "langchain", "python", "backend", "software engineer",
    "sde", "software developer", "data engineer", "platform",
    "computer vision", "research", "analytics", "algorithm",
    "intern", "internship", "trainee", "apprentice",
]


def _id(company: str, val: str) -> str:
    h = hashlib.md5(f"{company}_{val}".lower().encode()).hexdigest()[:8]
    return f"{company.lower()[:6]}_{h}"


def _is_relevant(title: str, desc: str = "") -> bool:
    combined = (title + " " + desc).lower()
    return any(kw in combined for kw in AI_KEYWORDS)


def _get(url: str, timeout: int = 20, extra_headers: dict = None) -> requests.Response | None:
    h = {**HEADERS, **(extra_headers or {})}
    try:
        r = requests.get(url, headers=h, timeout=timeout)
        if r.status_code == 200:
            return r
        log.debug(f"  HTTP {r.status_code}: {url[:80]}")
    except Exception as e:
        log.debug(f"  Error: {url[:60]} — {type(e).__name__}: {str(e)[:80]}")
    return None


# ── Greenhouse boards ─────────────────────────────────────────────────────────

def _fetch_greenhouse(company: str, board: str) -> list:
    jobs = []
    # Try both endpoints — some boards use v1, some use the public board URL
    urls = [
        f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true",
        f"https://boards.greenhouse.io/{board}/jobs.json",
    ]
    data = None
    for url in urls:
        r = _get(url)
        if r:
            try:
                data = r.json()
                break
            except Exception:
                continue

    if not data:
        return jobs

    job_list = data.get("jobs", data.get("postings", []))
    for j in job_list:
        title   = j.get("title", "")
        content = j.get("content", j.get("description", ""))
        if not _is_relevant(title, content):
            continue
        loc = j.get("location", {})
        if isinstance(loc, dict):
            loc = loc.get("name", "India")
        jobs.append({
            "job_id":      _id(board, str(j.get("id", title))),
            "title":       title,
            "company":     company,
            "location":    loc,
            "description": (content or "")[:1200],
            "url":         j.get("absolute_url", j.get("url", f"https://boards.greenhouse.io/{board}")),
            "posted":      (j.get("updated_at", j.get("created_at", "")) or "")[:10] or date.today().isoformat(),
            "source":      f"{company} Careers (Greenhouse)",
        })
    return jobs


def fetch_swiggy() -> list:
    jobs = _fetch_greenhouse("Swiggy", "swiggy")
    # Also try their direct careers page
    if not jobs:
        try:
            r = _get("https://careers.swiggy.com/api/fetch_jobs/?category=Technology")
            if r:
                for j in r.json().get("jobs", []):
                    title = j.get("title", "")
                    if not _is_relevant(title):
                        continue
                    jobs.append({
                        "job_id":      _id("swiggy", str(j.get("id", title))),
                        "title":       title,
                        "company":     "Swiggy",
                        "location":    j.get("location", "Bengaluru"),
                        "description": j.get("description", "")[:1200],
                        "url":         j.get("apply_url", "https://careers.swiggy.com"),
                        "posted":      date.today().isoformat(),
                        "source":      "Swiggy Careers",
                    })
        except Exception:
            pass
    log.info(f"Swiggy: {len(jobs)} jobs")
    return jobs


def fetch_meesho() -> list:
    jobs = _fetch_greenhouse("Meesho", "meesho")
    # Try direct API
    if not jobs:
        try:
            r = _get("https://meesho.io/jobs/api/jobs?department=Engineering&location=India")
            if r:
                data = r.json()
                for j in (data if isinstance(data, list) else data.get("jobs", [])):
                    title = j.get("title", j.get("name", ""))
                    if not _is_relevant(title):
                        continue
                    jobs.append({
                        "job_id":      _id("meesho", str(j.get("id", title))),
                        "title":       title,
                        "company":     "Meesho",
                        "location":    j.get("location", "Bengaluru"),
                        "description": j.get("description", "")[:1200],
                        "url":         j.get("url", j.get("applyUrl", "https://meesho.io/jobs")),
                        "posted":      date.today().isoformat(),
                        "source":      "Meesho Careers",
                    })
        except Exception:
            pass
    log.info(f"Meesho: {len(jobs)} jobs")
    return jobs


def fetch_atlassian() -> list:
    raw  = _fetch_greenhouse("Atlassian", "atlassian")
    jobs = [
        j for j in raw
        if any(w in j["location"].lower()
               for w in ["india", "remote", "bangalore", "hyderabad", "pune", "anywhere"])
    ]
    log.info(f"Atlassian: {len(jobs)} jobs")
    return jobs


# ── Lever boards ──────────────────────────────────────────────────────────────

def _fetch_lever(company: str, slug: str) -> list:
    jobs = []
    urls = [
        f"https://api.lever.co/v0/postings/{slug}?mode=json",
        f"https://jobs.lever.co/{slug}",
    ]
    for url in urls:
        r = _get(url)
        if not r:
            continue
        try:
            data = r.json()
            items = data if isinstance(data, list) else data.get("postings", [])
            for j in items:
                title = j.get("text", j.get("title", ""))
                if not _is_relevant(title, j.get("description", j.get("descriptionPlain", ""))):
                    continue
                cats = j.get("categories", {})
                jobs.append({
                    "job_id":      _id(slug, j.get("id", title)),
                    "title":       title,
                    "company":     company,
                    "location":    cats.get("location", j.get("location", "India")),
                    "description": (j.get("description", j.get("descriptionPlain", "")) or "")[:1200],
                    "url":         j.get("hostedUrl", j.get("applyUrl", f"https://jobs.lever.co/{slug}")),
                    "posted":      date.today().isoformat(),
                    "source":      f"{company} Careers (Lever)",
                })
            if jobs:
                break
        except Exception:
            continue
    return jobs


def fetch_razorpay() -> list:
    jobs = _fetch_lever("Razorpay", "razorpay")
    log.info(f"Razorpay: {len(jobs)} jobs")
    return jobs


def fetch_phonepe() -> list:
    jobs = _fetch_lever("PhonePe", "phonepe")
    log.info(f"PhonePe: {len(jobs)} jobs")
    return jobs


# ── Amazon ────────────────────────────────────────────────────────────────────

def fetch_amazon() -> list:
    """Amazon India entry-level jobs."""
    jobs = []
    queries = [
        ("Machine Learning", "ENTRY_LEVEL"),
        ("Software Development Engineer", "ENTRY_LEVEL"),
        ("Data Scientist", "ENTRY_LEVEL"),
        ("AI Engineer", "ENTRY_LEVEL"),
        ("Software Engineer", "ENTRY_LEVEL"),
        ("Machine Learning Intern", "INTERNSHIP"),
        ("Software Engineer Intern", "INTERNSHIP"),
        ("AI Intern", "INTERNSHIP"),
        ("Data Science Intern", "INTERNSHIP"),
    ]
    try:
        for q, level in queries:
            url = (
                "https://www.amazon.jobs/en/search.json"
                f"?base_query={requests.utils.quote(q)}"
                f"&loc_query=India&experience_ids={level}&result_limit=20"
            )
            r = _get(url)
            if not r:
                continue
            for j in r.json().get("jobs", []):
                title = j.get("title", "")
                desc  = j.get("description_short", "")
                if not _is_relevant(title, desc):
                    continue
                jobs.append({
                    "job_id":      _id("amazon", title + str(j.get("id_icims", ""))),
                    "title":       title,
                    "company":     "Amazon",
                    "location":    j.get("location", "India"),
                    "description": desc[:1200],
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

def fetch_google() -> list:
    """Google India entry-level jobs."""
    jobs = []
    try:
        for q in ["Machine Learning Engineer", "Software Engineer", "AI Research",
                   "Machine Learning Intern", "Software Engineer Intern", "AI Intern",
                   "Data Science Intern"]:
            url = (
                f"https://careers.google.com/api/v3/search/"
                f"?q={requests.utils.quote(q)}"
                "&location=India&jex=ENTRY_LEVEL&page_size=20&page=1"
            )
            r = _get(url)
            if not r:
                continue
            try:
                data = r.json()
            except Exception:
                continue
            for j in data.get("jobs", []):
                title = j.get("title", "")
                desc  = " ".join(
                    j.get("description", {}).get("responsibilities", []) +
                    j.get("description", {}).get("qualifications", [])
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

def fetch_microsoft() -> list:
    """Microsoft India — updated API endpoint June 2026."""
    jobs = []
    try:
        for q in ["AI Engineer", "Software Engineer", "Machine Learning",
                   "Software Engineer Intern", "AI Intern", "Machine Learning Intern"]:
            # Primary endpoint
            url = (
                "https://jobs.careers.microsoft.com/global/en/search"
                f"?q={requests.utils.quote(q)}"
                "&lc=India&pgSz=20&pg=1&format=json"
            )
            r = _get(url, extra_headers={"Referer": "https://jobs.careers.microsoft.com/"})
            if r:
                try:
                    data     = r.json()
                    job_list = (
                        data.get("operationResult", {})
                            .get("result", {})
                            .get("jobs", [])
                    )
                    if not job_list:
                        job_list = data.get("jobs", [])
                    for j in job_list:
                        title = j.get("title", j.get("Title", ""))
                        desc  = j.get("descriptionTeaser", j.get("Description", ""))
                        if not _is_relevant(title, desc):
                            continue
                        jid = j.get("jobId", j.get("JobId", ""))
                        jobs.append({
                            "job_id":      _id("microsoft", str(jid or title)),
                            "title":       title,
                            "company":     "Microsoft",
                            "location":    j.get("location", j.get("Location", "India")),
                            "description": desc[:1200],
                            "url":         f"https://jobs.careers.microsoft.com/global/en/job/{jid}",
                            "posted":      (j.get("postingDate", j.get("PostedDate", "")) or "")[:10],
                            "source":      "Microsoft Careers",
                        })
                except Exception as parse_err:
                    log.debug(f"Microsoft parse error: {parse_err}")
            time.sleep(1.2)
    except Exception as e:
        log.error(f"Microsoft error: {e}")
    log.info(f"Microsoft: {len(jobs)} jobs")
    return jobs


# ── Flipkart ──────────────────────────────────────────────────────────────────

def fetch_flipkart() -> list:
    """Flipkart careers — JSON-LD scrape from their jobs page."""
    jobs = []
    try:
        urls = [
            "https://www.flipkartcareers.com/#!/joblist",
            "https://www.flipkartcareers.com/#!/joblist?keyword=machine%20learning",
            "https://www.flipkartcareers.com/#!/joblist?keyword=software%20engineer",
        ]
        for url in urls:
            r = _get(url, timeout=25)
            if not r:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data  = json.loads(script.string)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if item.get("@type") not in ("JobPosting", "jobPosting"):
                            continue
                        title = item.get("title", "")
                        desc  = item.get("description", "")
                        if not _is_relevant(title, desc):
                            continue
                        org  = item.get("hiringOrganization", {})
                        loc  = item.get("jobLocation", {})
                        addr = loc.get("address", {}) if isinstance(loc, dict) else {}
                        jobs.append({
                            "job_id":      _id("flipkart", title + item.get("url", "")),
                            "title":       title,
                            "company":     org.get("name", "Flipkart") if isinstance(org, dict) else "Flipkart",
                            "location":    addr.get("addressLocality", "Bengaluru"),
                            "description": desc[:1200],
                            "url":         item.get("url", url),
                            "posted":      (item.get("datePosted") or "")[:10] or date.today().isoformat(),
                            "source":      "Flipkart Careers",
                        })
                except Exception:
                    pass
            time.sleep(1)
    except Exception as e:
        log.error(f"Flipkart error: {e}")
    log.info(f"Flipkart: {len(jobs)} jobs")
    return jobs


# ── Walmart ───────────────────────────────────────────────────────────────────

def fetch_walmart() -> list:
    """Walmart Global Tech India."""
    jobs = []
    try:
        for q in ["Machine Learning", "Software Engineer", "AI"]:
            url = (
                f"https://careers.walmart.com/api/jobs"
                f"?q={requests.utils.quote(q)}&location=India&page=1&pageSize=20"
            )
            r = _get(url, extra_headers={"Referer": "https://careers.walmart.com/"})
            if not r:
                continue
            try:
                data  = r.json()
                items = data.get("jobs", data.get("data", []))
                if not items and isinstance(data, list):
                    items = data
                for j in items:
                    title = j.get("title", j.get("jobTitle", ""))
                    desc  = j.get("shortDescription", j.get("description", ""))
                    if not _is_relevant(title, desc):
                        continue
                    jid = j.get("jobId", j.get("id", ""))
                    jobs.append({
                        "job_id":      _id("walmart", str(jid or title)),
                        "title":       title,
                        "company":     "Walmart Global Tech",
                        "location":    j.get("location", "Bengaluru"),
                        "description": desc[:1200],
                        "url":         f"https://careers.walmart.com/us/jobs/{jid}",
                        "posted":      (j.get("postedDate") or "")[:10] or date.today().isoformat(),
                        "source":      "Walmart Careers",
                    })
            except Exception:
                pass
            time.sleep(1)
    except Exception as e:
        log.error(f"Walmart error: {e}")
    log.info(f"Walmart: {len(jobs)} jobs")
    return jobs


# ── Adobe ─────────────────────────────────────────────────────────────────────

def fetch_adobe() -> list:
    """Adobe IDC India — Workday API."""
    jobs = []
    try:
        for q in ["Machine Learning", "AI Engineer", "Software Engineer", "Data"]:
            url = (
                "https://adobe.wd5.myworkdayjobs.com/wday/cxs/adobe/ADC/jobs"
                f"?q={requests.utils.quote(q)}&locations=India"
            )
            r = _get(url, extra_headers={"X-Requested-With": "XMLHttpRequest"})
            if not r:
                continue
            try:
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
            except Exception:
                pass
            time.sleep(1)
    except Exception as e:
        log.error(f"Adobe error: {e}")
    log.info(f"Adobe: {len(jobs)} jobs")
    return jobs


# ── Instahyre ─────────────────────────────────────────────────────────────────

def fetch_instahyre() -> list:
    """Instahyre India — AI-powered job platform."""
    jobs = []
    try:
        for kw in ["machine learning", "generative ai", "software engineer", "ai engineer", "python"]:
            url = f"https://www.instahyre.com/api/v1/job/?q={requests.utils.quote(kw)}&limit=30"
            r   = _get(url)
            if not r:
                continue
            try:
                data  = r.json()
                items = data if isinstance(data, list) else data.get("results", data.get("jobs", []))
                for j in items:
                    title       = j.get("designation", j.get("title", ""))
                    company_obj = j.get("company", {})
                    company     = company_obj.get("name", "") if isinstance(company_obj, dict) else j.get("company_name", "Unknown")
                    if not _is_relevant(title, j.get("description", "")):
                        continue
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
            except Exception:
                pass
            time.sleep(1)
    except Exception as e:
        log.error(f"Instahyre error: {e}")
    log.info(f"Instahyre: {len(jobs)} jobs")
    return jobs


# ── Wellfound ─────────────────────────────────────────────────────────────────

def fetch_wellfound() -> list:
    """Wellfound (AngelList) — startup jobs India."""
    jobs = []
    try:
        urls = [
            "https://wellfound.com/jobs/l/india/r/machine-learning-engineer",
            "https://wellfound.com/jobs/l/india/r/software-engineer",
            "https://wellfound.com/jobs/l/india/r/backend-engineer",
            "https://wellfound.com/jobs/l/india/r/data-scientist",
        ]
        for url in urls:
            r = _get(url, timeout=25, extra_headers={
                "Referer": "https://wellfound.com/",
                "Accept":  "text/html,application/xhtml+xml",
            })
            if not r:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data  = json.loads(script.string)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if item.get("@type") != "JobPosting":
                            continue
                        title = item.get("title", "")
                        desc  = item.get("description", "")
                        if not _is_relevant(title, desc):
                            continue
                        org  = item.get("hiringOrganization", {})
                        loc  = item.get("jobLocation", {})
                        addr = loc.get("address", {}) if isinstance(loc, dict) else {}
                        jobs.append({
                            "job_id":      _id("wellfound", title + item.get("url", "")),
                            "title":       title,
                            "company":     org.get("name", "Startup") if isinstance(org, dict) else "Startup",
                            "location":    addr.get("addressLocality", "India"),
                            "description": desc[:1200],
                            "url":         item.get("url", url),
                            "posted":      (item.get("datePosted") or "")[:10] or date.today().isoformat(),
                            "source":      "Wellfound",
                        })
                except Exception:
                    pass
            time.sleep(2)
    except Exception as e:
        log.error(f"Wellfound error: {e}")
    log.info(f"Wellfound: {len(jobs)} jobs")
    return jobs


# ── Internshala ───────────────────────────────────────────────────────────────

def fetch_internshala() -> list:
    """Internshala — fresher jobs and internships India."""
    jobs = []
    try:
        urls = [
            "https://internshala.com/internships/machine-learning-internship/",
            "https://internshala.com/internships/artificial-intelligence-internship/",
            "https://internshala.com/internships/python-internship/",
            "https://internshala.com/internships/data-science-internship/",
            "https://internshala.com/jobs/software-development-job/",
            "https://internshala.com/jobs/machine-learning-job/",
        ]
        for url in urls:
            r = _get(url, timeout=20)
            if not r:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data  = json.loads(script.string)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if item.get("@type") not in ("JobPosting", "Internship"):
                            continue
                        title = item.get("title", "")
                        if not title:
                            continue
                        org  = item.get("hiringOrganization", {})
                        loc  = item.get("jobLocation", {})
                        addr = loc.get("address", {}) if isinstance(loc, dict) else {}
                        jobs.append({
                            "job_id":      _id("internshala", title + str(item.get("url", ""))),
                            "title":       title,
                            "company":     org.get("name", "Company") if isinstance(org, dict) else "Company",
                            "location":    addr.get("addressLocality", "India"),
                            "description": (item.get("description") or "")[:1200],
                            "url":         item.get("url", url),
                            "posted":      (item.get("datePosted") or "")[:10] or date.today().isoformat(),
                            "source":      "Internshala",
                        })
                except Exception:
                    pass
            time.sleep(1.5)
    except Exception as e:
        log.error(f"Internshala error: {e}")
    log.info(f"Internshala: {len(jobs)} jobs")
    return jobs


# ── YCombinator ───────────────────────────────────────────────────────────────

def fetch_ycombinator() -> list:
    """YC Work at a Startup — top funded AI startups."""
    jobs = []
    try:
        url = "https://www.workatastartup.com/jobs.json?q=machine+learning+india&remote=true"
        r   = _get(url, timeout=15)
        if r:
            try:
                data = r.json()
                for j in (data if isinstance(data, list) else data.get("jobs", [])):
                    title = j.get("title", "")
                    desc  = j.get("description", "")
                    if not _is_relevant(title, desc):
                        continue
                    locs = j.get("locations", ["Remote"])
                    jobs.append({
                        "job_id":      _id("yc", str(j.get("id", title))),
                        "title":       title,
                        "company":     j.get("company", {}).get("name", "YC Startup") if isinstance(j.get("company"), dict) else "YC Startup",
                        "location":    locs[0] if locs else "Remote",
                        "description": desc[:1200],
                        "url":         f"https://www.workatastartup.com/jobs/{j.get('id','')}",
                        "posted":      date.today().isoformat(),
                        "source":      "YC Work at a Startup",
                    })
            except Exception:
                pass
    except Exception as e:
        log.error(f"YCombinator error: {e}")
    log.info(f"YCombinator: {len(jobs)} jobs")
    return jobs


# ── Naukri direct API ─────────────────────────────────────────────────────────

def fetch_naukri_direct() -> list:
    """Naukri direct search API — fresher tech roles India."""
    jobs = []
    try:
        naukri_headers = {
            "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept":          "application/json",
            "appid":           "109",
            "systemid":        "109",
            "naukri-platform": "desktop",
        }
        queries = [
            "generative ai fresher",
            "machine learning engineer fresher",
            "software engineer fresher 2027",
            "ai engineer fresher",
            "backend developer fresher python",
            "data scientist fresher",
        ]
        for q in queries:
            url = (
                "https://www.naukri.com/jobapi/v3/search"
                f"?noOfResults=20&urlType=search_by_keyword&searchType=adv"
                f"&keyword={requests.utils.quote(q)}&experience=0&location=India"
            )
            r = requests.get(url, headers=naukri_headers, timeout=15)
            if r.status_code != 200:
                continue
            try:
                data = r.json()
                for j in data.get("jobDetails", []):
                    title = j.get("title", "")
                    desc  = j.get("jobDescription", "")
                    if not _is_relevant(title, desc):
                        continue
                    placeholders = j.get("placeholders", [])
                    location     = placeholders[0].get("label", "India") if placeholders else "India"
                    jobs.append({
                        "job_id":      _id("naukri", str(j.get("jobId", title))),
                        "title":       title,
                        "company":     j.get("companyName", "Unknown"),
                        "location":    location,
                        "description": desc[:1200],
                        "url":         j.get("jdURL", "https://www.naukri.com"),
                        "posted":      date.today().isoformat(),
                        "source":      "Naukri",
                    })
            except Exception:
                pass
            time.sleep(1)
    except Exception as e:
        log.error(f"Naukri direct error: {e}")
    log.info(f"Naukri direct: {len(jobs)} jobs")
    return jobs


# ── Oracle ────────────────────────────────────────────────────────────────────

def fetch_oracle() -> list:
    """Oracle India careers."""
    jobs = []
    try:
        url = (
            "https://eeho.fa.us2.oraclecloud.com/hcmRestApi/resources/latest/"
            "recruitingCEJobRequisitions?onlyData=true"
            "&expand=requisitionList.secondaryLocations"
            "&finder=findReqs;siteNumber=CX_1,locationId=300000001201432&limit=30"
        )
        r = _get(url, timeout=20)
        if r:
            for j in r.json().get("items", []):
                for req in j.get("requisitionList", []):
                    title = req.get("Title", "")
                    if not _is_relevant(title):
                        continue
                    jobs.append({
                        "job_id":      _id("oracle", str(req.get("Id", title))),
                        "title":       title,
                        "company":     "Oracle",
                        "location":    req.get("PrimaryLocation", "India"),
                        "description": req.get("ShortDescriptionStr", "")[:1200],
                        "url":         f"https://careers.oracle.com/jobs/#en/sites/jobsearch/requisitions/{req.get('Id','')}",
                        "posted":      date.today().isoformat(),
                        "source":      "Oracle Careers",
                    })
    except Exception as e:
        log.error(f"Oracle error: {e}")
    log.info(f"Oracle: {len(jobs)} jobs")
    return jobs


# ── JPMorgan ──────────────────────────────────────────────────────────────────

def fetch_jpmorgan() -> list:
    """JPMorgan Chase India careers."""
    jobs = []
    try:
        url = (
            "https://jpmc.fa.oraclecloud.com/hcmRestApi/resources/latest/"
            "recruitingCEJobRequisitions?onlyData=true"
            "&finder=findReqs;siteNumber=CX_1,locationId=300000001506152&limit=30"
        )
        r = _get(url, timeout=20)
        if r:
            for j in r.json().get("items", []):
                for req in j.get("requisitionList", []):
                    title = req.get("Title", "")
                    if not _is_relevant(title):
                        continue
                    jobs.append({
                        "job_id":      _id("jpmorgan", str(req.get("Id", title))),
                        "title":       title,
                        "company":     "JPMorgan Chase",
                        "location":    req.get("PrimaryLocation", "India"),
                        "description": req.get("ShortDescriptionStr", "")[:1200],
                        "url":         f"https://careers.jpmorgan.com/global/en/jobs/{req.get('Id','')}",
                        "posted":      date.today().isoformat(),
                        "source":      "JPMorgan Careers",
                    })
    except Exception as e:
        log.error(f"JPMorgan error: {e}")
    log.info(f"JPMorgan: {len(jobs)} jobs")
    return jobs


# ── Master fetcher ────────────────────────────────────────────────────────────

def fetch_all_career_portals() -> list:
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
        ("Wellfound",     fetch_wellfound),
        ("Internshala",   fetch_internshala),
        ("YCombinator",   fetch_ycombinator),
        ("Naukri direct", fetch_naukri_direct),
        ("Oracle",        fetch_oracle),
        ("JPMorgan",      fetch_jpmorgan),
    ]
    for name, fn in fetchers:
        try:
            result = fn()
            all_jobs.extend(result)
        except Exception as e:
            log.error(f"Portal error — {name}: {e}")
        time.sleep(0.5)

    log.info(f"Career portals total: {len(all_jobs)} jobs")
    return all_jobs