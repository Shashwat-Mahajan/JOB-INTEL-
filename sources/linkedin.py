"""
sources/linkedin.py
Fetches LinkedIn jobs via the public guest API. Anonymous requests only —
deliberately does NOT use your real LinkedIn session/cookies. Anonymous
scraping that gets rate-limited just means fewer results that run; using
your actual logged-in session for automation risks LinkedIn flagging your
real account, which is a much worse outcome than a quiet day of fewer jobs.

--- CHANGED in this pass ---
  - Keywords are profile-driven (via utils.get_search_keywords, passed in
    as `keywords`), with a small hardcoded fallback only if none supplied.
  - Uses sources/base.py's resilient_get/new_session for retries/backoff.
  - Explicit login-wall detection: if LinkedIn redirects to an authwall/
    login page, that keyword's fetch stops immediately instead of
    retrying — no amount of retrying fixes a wall, and retrying just adds
    more flagged requests.
  - Added jitter (randomized delays) between requests/keywords — reduces
    how mechanically uniform the request pattern looks, without crossing
    into authenticated/account-risk territory.
  - Per-keyword fresh session (cookie/connection state doesn't carry
    across keywords) to keep each search looking like an independent,
    smaller burst rather than one long continuous session.
"""

import logging
import re
import time
import random
from datetime import date
from bs4 import BeautifulSoup
from urllib.parse import parse_qs, urlparse

from sources.base import resilient_get, new_session, validate_job

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_FALLBACK_KEYWORDS = [
    "generative AI engineer",
    "LLM engineer",
    "ML engineer",
    "AI engineer intern",
    "machine learning engineer fresher",
]

_VIEW_RE               = re.compile(r"/jobs/view/(\d+)")
_CJI_RE                = re.compile(r"currentJobId=(\d+)")
_JOB_POSTING_RE        = re.compile(r"/jobs-guest/jobs/api/jobPosting/(\d+)")
_GENERIC_NUMERIC_ID_RE = re.compile(r"\b(\d{8,11})\b")

_SENIOR_RE = re.compile(
    r"\b(senior|sr\.?|staff|principal|director|vp|chief|head\s+of|lead)\b",
    re.IGNORECASE,
)
_ENTRY_RE = re.compile(
    r"\b(intern|fresher|junior|entry|trainee|graduate|new grad)\b",
    re.IGNORECASE,
)

MAX_DETAIL_FETCHES_PER_KEYWORD = 8


def _extract_job_id(href: str) -> str | None:
    if not href:
        return None
    parsed = urlparse(href)
    cji = parse_qs(parsed.query).get("currentJobId", [None])[0]
    if cji:
        return cji
    for pattern in (_VIEW_RE, _CJI_RE, _JOB_POSTING_RE):
        m = pattern.search(href)
        if m:
            return m.group(1)
    return None


def _extract_job_id_from_card(card, href: str) -> str | None:
    job_id = _extract_job_id(href)
    if job_id:
        return job_id

    for attr_name in ("data-entity-urn", "data-job-id", "data-id", "data-occludable-job-id"):
        val = card.get(attr_name, "")
        if val:
            m = _GENERIC_NUMERIC_ID_RE.search(str(val))
            if m:
                return m.group(1)
    for tag in card.find_all(attrs={"data-entity-urn": True}):
        m = _GENERIC_NUMERIC_ID_RE.search(tag.get("data-entity-urn", ""))
        if m:
            return m.group(1)
    for tag in card.find_all(attrs={"data-job-id": True}):
        m = _GENERIC_NUMERIC_ID_RE.search(tag.get("data-job-id", ""))
        if m:
            return m.group(1)
    for candidate in card.find_all("a", href=True):
        m = _GENERIC_NUMERIC_ID_RE.search(candidate["href"])
        if m:
            return m.group(1)
    return None


def _canonical_url(job_id: str) -> str:
    return f"https://www.linkedin.com/jobs/view/{job_id}/"


def _fetch_job_description(session, job_id: str) -> str:
    url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    r = resilient_get(session, url, timeout=(5, 12), max_attempts=1)
    if r is None:
        return ""
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        desc_el = (
            soup.find("div", class_="description__text")
            or soup.find("div", class_="show-more-less-html__markup")
        )
        return desc_el.get_text(separator=" ", strip=True)[:2000] if desc_el else ""
    except Exception as e:
        log.debug(f"LinkedIn detail parse failed for {job_id}: {e}")
        return ""


def _parse_one_card(card, loc_fallback: str) -> dict | None:
    title_el   = card.find("h3", class_="base-search-card__title")
    company_el = card.find("h4", class_="base-search-card__subtitle")
    loc_el     = card.find("span", class_="job-search-card__location")

    link_el = card.find("a", class_="base-card__full-link")
    if not link_el:
        for a in card.find_all("a", href=True):
            href = a.get("href", "")
            if "/jobs/view/" in href or "currentJobId=" in href:
                link_el = a
                break
    if not link_el:
        link_el = card.find("a", href=True)

    title = title_el.get_text(strip=True) if title_el else ""
    if not title:
        return None

    href   = link_el.get("href", "") if link_el else ""
    job_id = _extract_job_id_from_card(card, href)
    if not job_id:
        return None

    return {
        "job_id":               f"li_{job_id}",
        "_linkedin_numeric_id": job_id,
        "title":                title,
        "company":              company_el.get_text(strip=True) if company_el else "Unknown",
        "location":             loc_el.get_text(strip=True) if loc_el else loc_fallback,
        "description":          "",
        "url":                  _canonical_url(job_id),
        "posted":               date.today().isoformat(),
        "source":               "LinkedIn",
    }


def fetch_linkedin(keywords: list, location: str = "India") -> list:
    """
    Fetch LinkedIn jobs anonymously via the public guest API, using
    profile-derived keywords. No session cookies, no authentication —
    deliberately avoids tying scraping activity to a real account.
    """
    search_terms = keywords if keywords else _FALLBACK_KEYWORDS

    jobs: list[dict]   = []
    seen_ids: set[str] = set()
    dropped_no_link    = 0
    dropped_invalid    = 0

    for kw in search_terms:
        session = new_session(HEADERS)   # fresh session per keyword
        detail_fetches = 0

        for start in [0, 25]:
            q = kw.replace(" ", "%20")
            l = location.replace(" ", "%20").replace(",", "%2C")
            url = (
                "https://www.linkedin.com/jobs-guest/jobs/api/"
                "seeMoreJobPostings/search"
                f"?keywords={q}&location={l}"
                "&f_E=1,2,3"
                "&f_TPR=r604800"
                f"&start={start}"
            )

            r = resilient_get(session, url, timeout=(5, 20), max_attempts=2, backoff=8)
            if r is None:
                log.debug(f"LinkedIn: no response for '{kw}' start={start}")
                time.sleep(5)
                continue

            # Login wall / authwall redirect — stop this keyword, retrying won't help
            if "authwall" in r.url or "/login" in r.url:
                log.warning(
                    f"LinkedIn: hit login/auth wall for '{kw}' — stopping this keyword. "
                    f"Anonymous guest access is being challenged; this is expected to "
                    f"happen intermittently and resolves on its own over time."
                )
                break

            try:
                soup  = BeautifulSoup(r.text, "html.parser")
                cards = soup.find_all("div", class_="base-card") or soup.find_all("li")

                for card in cards:
                    title_present = bool(card.find("h3", class_="base-search-card__title"))
                    job = _parse_one_card(card, location)
                    if job is None:
                        if title_present:
                            dropped_no_link += 1
                        continue

                    raw_id = job["job_id"]
                    if raw_id in seen_ids:
                        continue
                    seen_ids.add(raw_id)

                    is_senior = bool(_SENIOR_RE.search(job["title"]))
                    is_entry  = bool(_ENTRY_RE.search(job["title"]))

                    if not (is_senior and not is_entry) and detail_fetches < MAX_DETAIL_FETCHES_PER_KEYWORD:
                        numeric_id = job.pop("_linkedin_numeric_id")
                        desc = _fetch_job_description(session, numeric_id)
                        job["description"] = desc or "(Description unavailable — see URL)"
                        detail_fetches += 1
                        time.sleep(random.uniform(1.5, 3.0))
                    else:
                        job.pop("_linkedin_numeric_id", None)
                        job["description"] = "(Description unavailable — see URL)"

                    if not validate_job(job):
                        dropped_invalid += 1
                        continue

                    jobs.append(job)

            except Exception as e:
                log.error(f"LinkedIn parse error ('{kw}'): {e}")

            time.sleep(random.uniform(6, 10))

        time.sleep(random.uniform(10, 18))

    described = sum(1 for j in jobs if j["description"] and "unavailable" not in j["description"])
    log.info(
        f"LinkedIn: {len(jobs)} jobs ({described} with descriptions), "
        f"{dropped_no_link} dropped (no link), {dropped_invalid} invalid"
    )
    return jobs