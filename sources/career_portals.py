"""
sources/career_portals.py — Updated June 2026
Fixed endpoints for Microsoft, Flipkart, Walmart, Naukri.
Added Wellfound, Internshala, and YCombinator jobs for better fresher coverage.
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
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

AI_KEYWORDS = [
    "ai", "ml", "machine learning", "generative", "llm", "nlp",
    "deep learning", "data science", "artificial intelligence",
    "genai", "langchain", "python", "backend", "software engineer",
    "sde", "software developer", "data engineer", "platform engineer",
    "computer vision", "research", "analytics",
]

FRESHER_SIGNALS = [
    "fresher", "fresh graduate", "entry level", "entry-level",
    "0-1", "0-2", "0 - 1", "0 - 2", "new grad", "campus",
    "graduate", "2025", "2026", "2027", "trainee", "associate",
    "junior", "intern", "internship", "ppo",
]


def _id(company: str, val: str) -> str:
    h = hashlib.md5(f"{company}_{val}".lower().encode()).hexdigest()[:8]
    return f"{company.lower()[:6]}_{h}"


def _is_relevant(title: str, desc: str = "") -> bool:
    combined = (title + " " + desc).lower()
    return any(kw in combined for kw in AI_KEYWORDS)


def _get(url: str, timeout: int = 20, headers: dict = None) -> requests.Response | None:
    try:
        r = requests.get(url, headers=headers or HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r
        log.debug(f"  {url[:70]} → HTTP {r.status_code}")
    except Exception as e:
        log.debug(f"  {url[:70]} → {e}")
    return None


# ── Greenhouse boards ─────────────────────────────────────────────────────────

def _fetch_greenhouse(company: str, board: str) -> list:
    jobs = []
    r = _get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true")
    if not r:
        return jobs
    try:
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
    except Exception as e:
        log.error(f"{company} parse error: {e}")
    return jobs


def fetch_swiggy() -> list:
    jobs = _fetch_greenhouse("Swiggy", "swiggy")
    log.info(f"Swiggy: {len(jobs)} jobs")
    return jobs


def fetch_meesho() -> list:
    jobs = _fetch_greenhouse("Meesho", "meesho")
    log.info(f"Meesho: {len(jobs)} jobs")
    return jobs


def fetch_atlassian() -> list:
    raw  = _fetch_greenhouse("Atlassian", "atlassian")
    jobs = [j for j in raw if any(
        w in j["location"].lower()
        for w in ["india", "remote", "bangalore", "hyderabad", "pune"]
    )]
    log.info(f"Atlassian: {len(jobs)} jobs")
    return jobs


def fetch_walmart() -> list:
    """Walmart Global Tech India — fixed endpoint."""
    jobs = []
    try:
        # Updated Walmart API endpoint
        for q in ["Machine Learning", "Software Engineer", "AI", "Data"]:
            url = (
                f"https://careers.walmart.com/api/jobs"
                f"?q={requests.utils.quote(q)}"
                f"&location=Bengaluru%2C+India"
                f"&page=1&pageSize=20"
            )
            r = _get(url)
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
                    jobs.append({
                        "job_id":      _id("walmart", str(j.get("jobId", j.get("id", title)))),
                        "title":       title,
                        "company":     "Walmart Global Tech",
                        "location":    j.get("location", "Bengaluru"),
                        "description": desc[:1200],
                        "url":         f"https://careers.walmart.com/us/jobs/{j.get('jobId',j.get('id',''))}",
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


# ── Lever boards ──────────────────────────────────────────────────────────────

def _fetch_lever(company: str, slug: str) -> list:
    jobs = []
    r = _get(f"https://api.lever.co/v0/postings/{slug}?mode=json")
    if not r:
        return jobs
    try:
        for j in r.json():
            title = j.get("text", "")
            if not _is_relevant(title, j.get("description", "")):
                continue
            jobs.append({
                "job_id":      _id(slug, j.get("id", title)),
                "title":       title,
                "company":     company,
                "location":    j.get("categories", {}).get("location", "India"),
                "description": (j.get("description") or "")[:1200],
                "url":         j.get("hostedUrl", f"https://jobs.lever.co/{slug}"),
                "posted":      date.today().isoformat(),
                "source":      f"{company} Careers",
            })
    except Exception as e:
        log.error(f"{company} Lever error: {e}")
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
    """Amazon India — explicitly target entry-level roles."""
    jobs = []
    try:
        searches = [
            ("Machine Learning", "ENTRY_LEVEL"),
            ("Software Development Engineer", "ENTRY_LEVEL"),
            ("AI Engineer", "ENTRY_LEVEL"),
            ("Data Scientist", "ENTRY_LEVEL"),
            ("Software Engineer", "ENTRY_LEVEL"),
        ]
        for q, level in searches:
            url = (
                "https://www.amazon.jobs/en/search.json"
                f"?base_query={requests.utils.quote(q)}"
                f"&loc_query=India"
                f"&experience_ids={level}"
                f"&result_limit=20"
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
            time.sleep(1)
    except Exception as e:
        log.error(f"Google error: {e}")
    log.info(f"Google: {len(jobs)} jobs")
    return jobs


# ── Microsoft — fixed endpoint ────────────────────────────────────────────────

def fetch_microsoft() -> list:
    """Microsoft India careers — updated API endpoint."""
    jobs = []
    try:
        for q in ["AI Engineer", "Software Engineer", "Machine Learning"]:
            # Updated Microsoft careers API
            url = (
                f"https://jobs.careers.microsoft.com/global/en/search"
                f"?q={requests.utils.quote(q)}"
                f"&lc=India"
                f"&pgSz=20&pg=1&format=json"
            )
            r = _get(url)
            if not r:
                # Try alternate endpoint
                url2 = (
                    f"https://careers.microsoft.com/us/en/search-results"
                    f"?keywords={requests.utils.quote(q)}"
                    f"&location=India"
                )
                r = _get(url2)
                if not r:
                    continue
            try:
                data = r.json()
                job_list = (
                    data.get("operationResult", {})
                        .get("result", {})
                        .get("jobs", [])
                )
                if not job_list:
                    job_list = data.get("jobs", data.get("data", []))
                for j in job_list:
                    title = j.get("title", j.get("Title", ""))
                    desc  = j.get("descriptionTeaser", j.get("description", ""))
                    if not _is_relevant(title, desc):
                        continue
                    job_id = j.get("jobId", j.get("JobId", title))
                    jobs.append({
                        "job_id":      _id("microsoft", str(job_id)),
                        "title":       title,
                        "company":     "Microsoft",
                        "location":    j.get("location", j.get("Location", "India")),
                        "description": desc[:1200],
                        "url":         f"https://jobs.careers.microsoft.com/global/en/job/{job_id}",
                        "posted":      (j.get("postingDate", j.get("PostedDate", "")) or "")[:10],
                        "source":      "Microsoft Careers",
                    })
            except Exception:
                pass
            time.sleep(1)
    except Exception as e:
        log.error(f"Microsoft error: {e}")
    log.info(f"Microsoft: {len(jobs)} jobs")
    return jobs


# ── Flipkart — fixed endpoint ─────────────────────────────────────────────────

def fetch_flipkart() -> list:
    """Flipkart careers — updated scraping approach."""
    jobs = []
    try:
        # Try their workday-based API
        searches = ["machine learning", "software engineer", "data science", "backend"]
        for q in searches:
            url = (
                f"https://www.flipkartcareers.com/#!/joblist"
                f"?keyword={requests.utils.quote(q)}"
            )
            r = _get(url)
            if r:
                soup = BeautifulSoup(r.text, "html.parser")
                # Extract JSON-LD if present
                for script in soup.find_all("script", type="application/ld+json"):
                    try:
                        data  = json.loads(script.string)
                        items = data if isinstance(data, list) else [data]
                        for item in items:
                            if item.get("@type") != "JobPosting":
                                continue
                            title = item.get("title", "")
                            if not _is_relevant(title, item.get("description", "")):
                                continue
                            jobs.append({
                                "job_id":      _id("flipkart", title + item.get("url", "")),
                                "title":       title,
                                "company":     "Flipkart",
                                "location":    str(item.get("jobLocation", {}).get("address", {}).get("addressLocality", "Bengaluru")),
                                "description": (item.get("description") or "")[:1200],
                                "url":         item.get("url", "https://www.flipkartcareers.com"),
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


# ── Adobe ─────────────────────────────────────────────────────────────────────

def fetch_adobe() -> list:
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

def fetch_instahyre() -> list:
    jobs = []
    try:
        for kw in ["machine learning", "generative ai", "software engineer", "ai engineer"]:
            url = f"https://www.instahyre.com/api/v1/job/?q={requests.utils.quote(kw)}&limit=30"
            r   = _get(url)
            if not r:
                continue
            data  = r.json()
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


# ── NEW: Wellfound (AngelList) — great for AI startups ───────────────────────

def fetch_wellfound() -> list:
    """Wellfound — best source for funded AI startups hiring freshers."""
    jobs = []
    try:
        searches = [
            "https://wellfound.com/jobs/l/india/r/machine-learning-engineer",
            "https://wellfound.com/jobs/l/india/r/software-engineer",
            "https://wellfound.com/jobs/l/india/r/backend-engineer",
            "https://wellfound.com/jobs/l/india/r/data-scientist",
        ]
        for url in searches:
            r = _get(url, timeout=25)
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


# ── NEW: Internshala — best for fresher + internship + PPO roles ─────────────

def fetch_internshala() -> list:
    """Internshala — huge source for fresher jobs and PPO internships in India."""
    jobs = []
    try:
        searches = [
            ("machine learning", "https://internshala.com/internships/machine-learning-internship/"),
            ("ai",               "https://internshala.com/internships/artificial-intelligence-internship/"),
            ("python",           "https://internshala.com/internships/python-internship/"),
            ("data science",     "https://internshala.com/internships/data-science-internship/"),
            ("backend",          "https://internshala.com/internships/web-development-internship/"),
            ("sde",              "https://internshala.com/jobs/software-development-job/"),
            ("ml engineer",      "https://internshala.com/jobs/machine-learning-job/"),
        ]
        for kw, url in searches:
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
                        desc  = item.get("description", "")
                        if not _is_relevant(title, desc):
                            continue
                        org  = item.get("hiringOrganization", {})
                        loc  = item.get("jobLocation", {})
                        addr = loc.get("address", {}) if isinstance(loc, dict) else {}
                        jobs.append({
                            "job_id":      _id("internshala", title + str(item.get("url", ""))),
                            "title":       title,
                            "company":     org.get("name", "Company") if isinstance(org, dict) else "Company",
                            "location":    addr.get("addressLocality", "India"),
                            "description": desc[:1200],
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


# ── NEW: YCombinator jobs — top AI startups ───────────────────────────────────

def fetch_ycombinator() -> list:
    """YC Work at a Startup — top funded AI startups, many hire in India/remote."""
    jobs = []
    try:
        url = "https://www.workatastartup.com/jobs.json?q=machine+learning+india&remote=true"
        r   = _get(url, timeout=15)
        if r:
            data = r.json()
            for j in (data if isinstance(data, list) else data.get("jobs", [])):
                title = j.get("title", "")
                desc  = j.get("description", "")
                if not _is_relevant(title, desc):
                    continue
                jobs.append({
                    "job_id":      _id("yc", str(j.get("id", title))),
                    "title":       title,
                    "company":     j.get("company", {}).get("name", "YC Startup"),
                    "location":    j.get("locations", ["Remote"])[0] if j.get("locations") else "Remote",
                    "description": desc[:1200],
                    "url":         f"https://www.workatastartup.com/jobs/{j.get('id','')}",
                    "posted":      date.today().isoformat(),
                    "source":      "YC Work at a Startup",
                })
    except Exception as e:
        log.error(f"YCombinator error: {e}")
    log.info(f"YCombinator: {len(jobs)} jobs")
    return jobs


# ── NEW: Naukri direct search ─────────────────────────────────────────────────

def fetch_naukri_direct() -> list:
    """Naukri — direct API search for fresher tech roles."""
    jobs = []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "appid": "109",
            "systemid": "109",
        }
        searches = [
            "generative ai fresher",
            "generative ai intern",
            "machine learning engineer fresher",
            "machine learning intern",
            "software engineer fresher 2027",
            "software engineer intern",
            "ai engineer fresher",
            "ai engineer intern",
            "backend developer fresher python",
            "backend developer intern",
        ]
        for q in searches:
            url = (
                f"https://www.naukri.com/jobapi/v3/search"
                f"?noOfResults=20"
                f"&urlType=search_by_keyword"
                f"&searchType=adv"
                f"&keyword={requests.utils.quote(q)}"
                f"&experience=0"
                f"&location=India"
            )
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                continue
            try:
                data = r.json()
                for j in data.get("jobDetails", []):
                    title = j.get("title", "")
                    desc  = j.get("jobDescription", "")
                    if not _is_relevant(title, desc):
                        continue
                    jobs.append({
                        "job_id":      _id("naukri", str(j.get("jobId", title))),
                        "title":       title,
                        "company":     j.get("companyName", "Unknown"),
                        "location":    j.get("placeholders", [{}])[0].get("label", "India") if j.get("placeholders") else "India",
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


# ── Master fetcher ────────────────────────────────────────────────────────────

def fetch_all_career_portals() -> list:
    all_jobs = []
    fetchers = [
        ("Swiggy",       fetch_swiggy),
        ("Meesho",       fetch_meesho),
        ("Atlassian",    fetch_atlassian),
        ("Razorpay",     fetch_razorpay),
        ("PhonePe",      fetch_phonepe),
        ("Amazon",       fetch_amazon),
        ("Google",       fetch_google),
        ("Microsoft",    fetch_microsoft),
        ("Flipkart",     fetch_flipkart),
        ("Walmart",      fetch_walmart),
        ("Adobe",        fetch_adobe),
        ("Instahyre",    fetch_instahyre),
        ("Wellfound",    fetch_wellfound),
        ("Internshala",  fetch_internshala),
        ("YCombinator",  fetch_ycombinator),
        ("NaukriDirect", fetch_naukri_direct),
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