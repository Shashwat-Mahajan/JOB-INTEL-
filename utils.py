"""
utils.py — profile loading, intra-batch dedup, config helpers.
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

PROFILE_PATH = Path(__file__).parent / "config" / "profile.json"

def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_profile(path: Path | None = None) -> dict:
    """Load structured candidate profile from config/profile.json."""
    path = path or PROFILE_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load profile: {e}")
        return {}


def get_scoring_prompt(profile: dict | None = None) -> str:
    """Return the LLM scoring prompt built from the resume profile."""
    profile = profile if profile is not None else load_profile()
    prompt  = profile.get("_scoring_prompt", "")
    if prompt:
        return prompt

    name  = profile.get("name", "Candidate")
    batch = profile.get("graduation_batch", "2027")
    roles = ", ".join(profile.get("target_roles", ["GenAI Engineer", "ML Engineer"]))
    return (
        f"CANDIDATE: {name}\n"
        f"GRADUATION: {batch} batch | EXPERIENCE: fresher (0-2 years)\n"
        f"TARGET ROLES: {roles}\n"
        f"INCLUDE: full-time jobs AND paid internships (PPO preferred).\n"
        f"VETOES: IT outsourcing without AI/ML title, 3+ years required, unpaid internships."
    )


def get_verifier_system_prompt(profile: dict | None = None) -> str:
    """Build the second-pass verifier prompt from the live resume profile."""
    profile  = profile if profile is not None else load_profile()
    candidate = get_scoring_prompt(profile)
    batch    = profile.get("graduation_batch", "2027")

    return f"""You are a strict senior recruiter doing a SECOND OPINION on listings already scored HIGH.
Be MORE critical than the first pass. Judge authenticity — reject vague, recycled, or misleading posts.

CANDIDATE PROFILE (from resume):
{candidate}

SCOPE: Score BOTH full-time jobs AND internships. Paid internships and PPO roles are valid HIGH matches.

For each listing answer:
1. Does this GENUINELY involve AI/ML/backend/software engineering work (not keyword stuffing)?
2. Is the company a product company, AI startup, or reputable tech employer (not generic IT services)?
3. Is this truly open to someone graduating {batch} (fresher, intern, 0-2 years, campus, or explicit intern role)?

ALL THREE yes → keep HIGH.
Any doubt → downgrade to MEDIUM.
Clearly wrong, scam-like, unpaid (when veto applies), or off-profile → SKIP.

Return ONLY valid JSON array:
[{{"job_id":"...","verified_priority":"HIGH|MEDIUM|LOW|SKIP","confidence":0-100,"posting_type":"job|internship|unknown","reason":"one sentence"}}]
"""


def deduplicate_batch(jobs: list) -> list:
    """
    FIX 5: Cross-source dedup that catches same job posted across multiple
    sources (e.g. WebBoost appearing 3x from LinkedIn + Remotive + career portal).

    Old version keyed on job_id first, then fell back to title|company only
    when job_id was missing. Jobs from different sources get different job_ids,
    so the title|company fallback never fired — duplicates slipped through.

    New approach: always register both job_id and title|company as keys,
    pointing to the same record. A second encounter on either key deduplicates.
    Keeps the richest record (longest description) per logical role.
    """
    best_by_id:      dict[str, dict] = {}   # job_id  → job
    best_by_title:   dict[str, dict] = {}   # title|company → job

    def _title_key(job: dict) -> str:
        t = (job.get("title") or "").strip().lower()
        c = (job.get("company") or "").strip().lower()
        return f"{t}|{c}"

    def _is_richer(candidate: dict, existing: dict) -> bool:
        return len(candidate.get("description") or "") > len(existing.get("description") or "")

    for job in jobs:
        jid   = (job.get("job_id") or "").strip()
        tkey  = _title_key(job)
        valid_tkey = tkey != "|" and bool(tkey.replace("|", "").strip())

        # Check if we've already seen this job via title|company
        if valid_tkey and tkey in best_by_title:
            existing = best_by_title[tkey]
            if _is_richer(job, existing):
                # Replace with richer version; keep both index entries consistent
                best_by_title[tkey] = job
                if jid:
                    best_by_id[jid] = job
                old_jid = (existing.get("job_id") or "").strip()
                if old_jid and old_jid in best_by_id:
                    best_by_id[old_jid] = job
            continue  # already registered — skip re-registration

        # Check if we've seen this job_id before
        if jid and jid in best_by_id:
            existing = best_by_id[jid]
            if _is_richer(job, existing):
                best_by_id[jid] = job
                old_tkey = _title_key(existing)
                if old_tkey in best_by_title:
                    best_by_title[old_tkey] = job
            continue

        # New job — register under both keys
        if jid:
            best_by_id[jid] = job
        if valid_tkey:
            best_by_title[tkey] = job

    # Collect unique objects (both dicts may point to same object)
    seen_object_ids: set = set()
    result: list = []
    for job in list(best_by_id.values()) + list(best_by_title.values()):
        oid = id(job)
        if oid not in seen_object_ids:
            seen_object_ids.add(oid)
            result.append(job)

    return result


def load_seen(path: Path) -> dict:
    """
    Load seen jobs as {job_id: date_first_seen}.
    Automatically expires entries older than 7 days.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Handle old format (plain list) — migrate to dict
        if isinstance(data, list):
            today = date.today().isoformat()
            data  = {jid: today for jid in data}

        # Expire jobs older than 7 days
        cutoff  = (date.today() - timedelta(days=7)).isoformat()
        cleaned = {jid: seen_date for jid, seen_date in data.items()
                   if seen_date >= cutoff}

        expired = len(data) - len(cleaned)
        if expired > 0:
            logging.getLogger(__name__).info(
                f"Dedup: expired {expired} old entries (>7 days)"
            )
        return cleaned
    except Exception:
        return {}


def save_seen(path: Path, seen: dict) -> None:
    """Save seen jobs dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(seen, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def deduplicate(jobs: list, seen: dict) -> tuple:
    """
    Filter to only unseen jobs.
    Adds new job_ids to seen with today's date.
    Returns (fresh_jobs, updated_seen).
    """
    today = date.today().isoformat()
    fresh = []
    for job in jobs:
        jid = job.get("job_id", "")
        if jid and jid not in seen:
            fresh.append(job)
            seen[jid] = today
    return fresh, seen


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. "
            "Copy config/config.example.json → config/config.json"
        )
    return json.loads(path.read_text(encoding="utf-8"))

# ── Profile-driven search keywords ───────────────────────────────────────────
# Generates LinkedIn/Naukri search queries directly from profile.json's
# target_roles, so fetch-stage queries and scoring-stage judgment both stay
# anchored to the same resume source of truth instead of drifting apart
# (previously linkedin.py had its own hardcoded LINKEDIN_SEARCHES list that
# had no connection to profile.json at all).

_FALLBACK_SEARCH_KEYWORDS = [
    "generative AI engineer", "LLM engineer", "machine learning engineer",
    "AI engineer intern", "software engineer AI", "backend engineer python",
    "SDE fresher", "data science intern",
]

def get_search_keywords(profile: dict | None = None) -> list[str]:
    """
    Build search keywords from profile.json's target_roles.
    Falls back to a small hardcoded list if no profile/target_roles exist,
    so the system never has zero search terms.
    """
    profile = profile if profile is not None else load_profile()
    roles = profile.get("target_roles", [])
    if not roles:
        logging.getLogger(__name__).warning(
            "No target_roles in profile.json — using fallback search keywords"
        )
        return _FALLBACK_SEARCH_KEYWORDS

    keywords = []
    for role in roles:
        role = role.strip()
        if not role:
            continue
        keywords.append(role)
        keywords.append(f"{role} intern")
        keywords.append(f"{role} fresher")

    # de-dupe while preserving order
    seen = set()
    out = []
    for k in keywords:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            out.append(k)
    return out or _FALLBACK_SEARCH_KEYWORDS