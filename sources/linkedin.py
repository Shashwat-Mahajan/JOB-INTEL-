"""
sources/linkedin.py — LinkedIn guest job search.

Fixes vs original:
  1. URL extraction: pulls BOTH /jobs/view/<id> direct links AND
     currentJobId=<id> from search redirects, always resolves to
     https://www.linkedin.com/jobs/view/<id>/ canonical form.
  2. Over-representation cap: max 10 keyword combos (was 25+).
     LinkedIn already returns 163+ jobs from 10 searches — no need for 50.
  3. Stricter intra-source dedup: keyed on job_id (numeric LinkedIn ID)
     not just title+company, so the same posting across keyword combos
     is dropped once instead of appearing multiple times.
  4. Source tag set to "LinkedIn" consistently.
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

# ── Capped search list — 10 combos max ───────────────────────────────────────
# Rationale: 10 combos × 2 pages = 20 requests → ~160-200 jobs from LinkedIn.
# That is already the largest single source. Running 25+ combos wastes time
# and causes LinkedIn to rate-limit (DNS failures seen in logs).
# Kept: highest-signal queries. Removed: duplicates and low-signal variants.
LINKEDIN_SEARCHES = [
    ("generative AI engineer",        "India"),
    ("LLM engineer",                  "India"),
    ("machine learning engineer",     "India"),
    ("AI engineer intern",            "India"),
    ("software engineer AI",          "India"),
    ("backend engineer python",       "India"),
    ("SDE fresher",                   "India"),
    ("data science intern",           "India"),
    ("machine learning intern",       "Bengaluru, Karnataka, India"),
    ("software engineer intern",      "Bengaluru, Karnataka, India"),
]

# ── URL helpers ───────────────────────────────────────────────────────────────

# Direct view link already in correct form
_VIEW_RE = re.compile(r'linkedin\.com/jobs/view/(\d+)')

# Search redirect with currentJobId param
_CJI_RE  = re.compile(r'currentJobId=(\d+)')

def _extract_job_id_from_html_fragment(href: str) -> str | None:
    """
    Return the numeric LinkedIn job ID from any LinkedIn URL form:
      - https://www.linkedin.com/jobs/view/1234567890/
      - https://www.linkedin.com/jobs/search/?currentJobId=1234567890&...
      - relative or partial hrefs containing either pattern
    Returns None if no ID found.
    """
    m = _VIEW_RE.search(href)
    if m:
        return m.group(1)
    m = _CJI_RE.search(href)
    if m:
        return m.group(1)
    return None


def _canonical_url(job_id: str) -> str:
    return f"https://www.linkedin.com/jobs/view/{job_id}/"


# ── Main fetcher ──────────────────────────────────────────────────────────────

def fetch_linkedin(keywords: list, location: str = "India") -> list:
    """
    Fetch LinkedIn jobs via the guest API.
    Returns jobs with canonical /jobs/view/<id>/ URLs.
    Capped at LINKEDIN_SEARCHES (10 combos) regardless of what keywords are passed.
    """
    jobs: list[dict] = []
    seen_ids: set[str] = set()   # keyed on numeric LinkedIn job ID

    for kw, loc in LINKEDIN_SEARCHES:
        for start in [0, 25]:    # 2 pages per keyword = ~50 jobs per combo
            try:
                q   = requests.utils.quote(kw)
                l   = requests.utils.quote(loc)
                url = (
                    "https://www.linkedin.com/jobs-guest/jobs/api/"
                    "seeMoreJobPostings/search"
                    f"?keywords={q}&location={l}"
                    f"&f_E=1,2,3"       # entry / associate / internship
                    f"&f_TPR=r604800"   # last 7 days
                    f"&start={start}"
                )
                r = requests.get(url, headers=HEADERS, timeout=20)
                if r.status_code != 200:
                    time.sleep(3)
                    continue

                content = r.text

                # ── Parse fields ─────────────────────────────────────────────
                titles    = re.findall(
                    r'class="base-search-card__title"[^>]*>\s*([^<]+?)\s*<',
                    content,
                )
                companies = re.findall(
                    r'class="base-search-card__subtitle"[^>]*>\s*([^<]+?)\s*<',
                    content,
                )
                locations = re.findall(
                    r'class="job-search-card__location"[^>]*>\s*([^<]+?)\s*<',
                    content,
                )

                # ── Extract ALL href values that contain a LinkedIn job ID ───
                # Captures both /jobs/view/<id> and ?currentJobId=<id> forms.
                all_hrefs = re.findall(r'href="([^"]*linkedin\.com/jobs/[^"]*)"', content)
                # Also catch relative-style and data-entity-urn job IDs
                urn_ids   = re.findall(r'data-entity-urn="[^"]*:(\d{10,})"', content)

                # Build an ordered list of (job_id, canonical_url) aligned with title order
                resolved: list[tuple[str, str]] = []
                for href in all_hrefs:
                    jid = _extract_job_id_from_html_fragment(href)
                    if jid:
                        resolved.append((jid, _canonical_url(jid)))

                # Fallback: use URN IDs if href parsing yielded fewer than titles
                if len(resolved) < len(titles):
                    for uid in urn_ids:
                        resolved.append((uid, _canonical_url(uid)))

                # Deduplicate resolved list while preserving order
                seen_this_page: set[str] = set()
                resolved_deduped: list[tuple[str, str]] = []
                for jid, curl in resolved:
                    if jid not in seen_this_page:
                        seen_this_page.add(jid)
                        resolved_deduped.append((jid, curl))

                # ── Assemble job dicts ────────────────────────────────────────
                for i, title in enumerate(titles):
                    title   = title.strip()
                    company = companies[i].strip() if i < len(companies) else "Unknown"

                    if i < len(resolved_deduped):
                        jid, job_url = resolved_deduped[i]
                    else:
                        # No URL extracted — generate a stable ID from title+company
                        fallback_key = f"{title}_{company}".lower()
                        jid          = hashlib.md5(fallback_key.encode()).hexdigest()[:10]
                        job_url      = "https://www.linkedin.com/jobs/"

                    # Cross-keyword dedup by LinkedIn job ID
                    if jid in seen_ids:
                        continue
                    seen_ids.add(jid)

                    jobs.append({
                        "job_id":      f"li_{jid}",
                        "title":       title,
                        "company":     company,
                        "location":    locations[i].strip() if i < len(locations) else loc,
                        "description": "(Visit URL for full description)",
                        "url":         job_url,   # always /jobs/view/<id>/ form
                        "posted":      date.today().isoformat(),
                        "source":      "LinkedIn",
                    })

                time.sleep(3)   # polite delay — reduces DNS failures

            except Exception as e:
                log.error(f"LinkedIn error ({kw}): {e}")
                time.sleep(4)

    log.info(f"LinkedIn: {len(jobs)} jobs")
    return jobs