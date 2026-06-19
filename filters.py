"""
filters.py — Pre-LLM filters. Run AFTER dedup, BEFORE any Groq call.

Order inside apply_all_filters():
  1. LinkedIn URL fix        → /jobs/view/<id>/ direct links
  2. Job-type classification → job["job_type"] = "internship" | "full-time"
  3. Seniority drop          → senior/staff/lead/principal/manager/director …
  4. Experience filter       → drop > 2 years required (internships always pass)

Usage in crew.py fetch_all_jobs_tool:
    from filters import apply_all_filters
    fresh, seen = deduplicate(raw, seen)
    fresh = apply_all_filters(fresh)          # ← insert here
    _state["fresh_jobs"] = fresh
"""

from __future__ import annotations
import re
import logging
from typing import Any

log = logging.getLogger(__name__)

# ── 1. LinkedIn URL fix ───────────────────────────────────────────────────────

_LI_SEARCH_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/jobs/search/?\?[^\s\"']*currentJobId=(\d+)[^\s\"']*"
)
_LI_VIEW_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/jobs/view/(\d+)"
)

def fix_linkedin_url(url: str) -> str:
    """
    Convert any LinkedIn search URL with currentJobId=XYZ → /jobs/view/XYZ/
    Already-correct view URLs are normalised (trailing slash).
    Non-LinkedIn URLs pass through unchanged.
    """
    if not url:
        return url
    m = _LI_VIEW_RE.search(url)
    if m:
        return f"https://www.linkedin.com/jobs/view/{m.group(1)}/"
    m = _LI_SEARCH_RE.search(url)
    if m:
        return f"https://www.linkedin.com/jobs/view/{m.group(1)}/"
    return url


# ── 2. Job-type classifier ────────────────────────────────────────────────────

_INTERN_TITLE_RE = re.compile(
    r"\b(intern(ship)?|trainee|apprentice|co[-\s]?op|placement|"
    r"graduate\s+trainee|summer\s+analyst|summer\s+associate)\b",
    re.IGNORECASE,
)
_INTERN_TYPE_VALUES = {
    "internship", "intern", "apprenticeship", "trainee",
    "co-op", "contract to hire",
}

def classify_job_type(job: dict[str, Any]) -> str:
    """Return 'internship' or 'full-time'."""
    emp_type = (job.get("employment_type") or "").lower().strip()
    if emp_type in _INTERN_TYPE_VALUES:
        return "internship"
    if _INTERN_TITLE_RE.search(job.get("title", "")):
        return "internship"
    return "full-time"


# ── 3. Seniority filter ───────────────────────────────────────────────────────

_SENIORITY_RE = re.compile(
    r"\b("
    r"senior|sr\.?|staff|lead|principal|manager|director|"
    r"head\s+of|vp|vice\s+president|architect|chief|"
    r"engineering\s+manager|tech\s+lead|team\s+lead|em\b"
    r")\b",
    re.IGNORECASE,
)

def is_senior_role(title: str) -> bool:
    return bool(_SENIORITY_RE.search(title))


# ── 3b. Non-engineering "AI-adjacent" veto ────────────────────────────────────
# A naive keyword match on "ai"/"ml"/"genai" in AI_KEYWORDS (career_portals.py)
# lets through roles that mention AI but are NOT engineering work — e.g.
# "AI Content Writer", "AI Marketing Specialist", "AI Trainer (Sales)".
# This filter drops those before they ever reach the scorer, since the
# scorer's keyword-ish rubric can't reliably distinguish "AI" in a content/
# sales/ops title from "AI" in an actual engineering title.

_NON_ENGINEERING_TITLE_RE = re.compile(
    r"\b("
    r"content\s+writer|content\s+creator|copywriter|copy\s+writer|"
    r"content\s+strategist|content\s+marketing|"
    r"marketing\s+(specialist|manager|executive|associate)|"
    r"sales\s+(executive|representative|associate|manager)|"
    r"business\s+development|account\s+manager|"
    r"recruiter|talent\s+acquisition|hr\s+(executive|specialist|manager)|"
    r"customer\s+(success|support|service)|"
    r"trainer\b(?!.*\b(model|engineer)\b)|"   # "AI Trainer" without "model"/"engineer" nearby
    r"prompt\s+engineer\b.*\b(content|marketing|writer)\b|"
    r"community\s+manager|social\s+media|"
    r"voiceover|voice\s+over|"
    r"data\s+entry|data\s+annotat(or|ion)\s+(?!.*engineer)"   # pure annotation, not eng
    r")\b",
    re.IGNORECASE,
)

# Titles that legitimately combine AI + a non-eng word but ARE engineering
# (avoid false-vetoing these)
_ENGINEERING_OVERRIDE_RE = re.compile(
    r"\b(engineer|developer|scientist|researcher|architect|sde|"
    r"backend|full\s*stack|mlops|devops)\b",
    re.IGNORECASE,
)

def is_non_engineering_role(title: str) -> bool:
    """
    Return True if the title is an AI-adjacent but non-engineering role
    (content, marketing, sales, HR, support, etc).
    A clear engineering term anywhere in the title overrides the veto —
    e.g. "AI Content Engineer" is kept, "AI Content Writer" is dropped.
    """
    if _ENGINEERING_OVERRIDE_RE.search(title):
        return False
    return bool(_NON_ENGINEERING_TITLE_RE.search(title))


# ── 4. Experience extractor & filter ─────────────────────────────────────────

_EXP_PATTERNS = [
    re.compile(r"(\d+)\s*\+?\s*(?:to\s*\d+\s*)?years?\s+(?:of\s+)?experience", re.IGNORECASE),
    re.compile(r"minimum\s+(?:of\s+)?(\d+)\s+years?", re.IGNORECASE),
    re.compile(r"at\s+least\s+(\d+)\s+years?", re.IGNORECASE),
    re.compile(r"(\d+)\s*[-–]\s*\d+\s+years?\s+(?:of\s+)?experience", re.IGNORECASE),
    re.compile(r"experience\s*(?:of|:)?\s*(\d+)\s*\+?\s*years?", re.IGNORECASE),
    re.compile(r"(\d+)\s+yrs?\b", re.IGNORECASE),
]

def extract_required_experience(text: str) -> int | None:
    """Return minimum years required, or None if not found."""
    found: list[int] = []
    for pat in _EXP_PATTERNS:
        for m in pat.finditer(text):
            try:
                found.append(int(m.group(1)))
            except (IndexError, ValueError):
                pass
    return min(found) if found else None

_FRESHER_SIGNAL_RE = re.compile(
    r"\b(fresher|entry[\s-]?level|0[\s-]?to[\s-]?2|0[-\s]?2\s*years?|"
    r"early[\s-]?career|new\s+grad|graduate\s+(program|role)|junior)\b",
    re.IGNORECASE,
)

def passes_experience_filter(job: dict[str, Any]) -> bool:
    """
    Internships always pass.
    Explicit years > 2 → drop.
    Explicit years <= 2 → pass.
    No explicit years stated:
      - If title/description has an explicit fresher/entry-level signal → pass.
      - Otherwise → drop (was previously a blanket pass — too permissive,
        let through senior-leaning roles with vague descriptions).
    """
    if job.get("job_type") == "internship":
        return True

    text = f"{job.get('title', '')} {job.get('description', '')}"
    years = extract_required_experience(text)

    if years is not None:
        return years <= 2

    # No explicit years — require a positive fresher/entry signal instead
    # of defaulting to pass. This is the stricter behavior.
    return bool(_FRESHER_SIGNAL_RE.search(text))


# ── Master pipeline ───────────────────────────────────────────────────────────

def apply_all_filters(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Run every pre-LLM filter in a single pass.

    Mutates each job dict:
      job["url"]      → fixed LinkedIn direct link (all sources safe to call)
      job["job_type"] → "internship" | "full-time"

    Returns only jobs that pass seniority, experience, and role-type gates.
    Call this AFTER deduplicate(), BEFORE score_jobs_with_llm().
    """
    kept: list[dict[str, Any]] = []
    n_seniority = 0
    n_experience = 0
    n_non_engineering = 0

    for job in jobs:
        # 1. Fix URL (safe on all sources — non-LinkedIn URLs pass through)
        job["url"] = fix_linkedin_url(job.get("url", ""))

        # 2. Classify type first — so internships bypass seniority gate below
        job["job_type"] = classify_job_type(job)

        # 3. Non-engineering veto — drops "AI Content Writer", "AI Marketing
        #    Specialist" etc that matched on the bare word "AI" upstream but
        #    are not engineering roles. Checked before seniority/experience
        #    since it's a category mismatch, not a level mismatch.
        if is_non_engineering_role(job.get("title", "")):
            n_non_engineering += 1
            continue

        # 4. Seniority gate (internships skip this — a "Senior Intern" is still an intern)
        if job["job_type"] != "internship" and is_senior_role(job.get("title", "")):
            n_seniority += 1
            continue

        # 5. Experience gate
        if not passes_experience_filter(job):
            n_experience += 1
            continue

        kept.append(job)

    log.info(
        "Pre-LLM filter: %d in → %d out | dropped non_engineering=%d seniority=%d experience=%d",
        len(jobs), len(kept), n_seniority, n_experience,
    )
    return kept