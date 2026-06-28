"""
scorer.py — 4-layer hybrid scoring pipeline.

v2.2 changes vs v2.1:
  - SCORING_SYSTEM replaced with _build_scoring_system(profile) — no hardcoded AI/ML axes
  - _build_profile_anchor() now builds from CANDIDATE_PROFILE instead of hardcoded AI/ML terms
  - _build_ce_query() now builds from CANDIDATE_PROFILE instead of hardcoded AI/ML terms
  - _VERIFIER_SYSTEM updated to be profile-driven
  - _layer2_nim_score and _score_batch_with_retries accept scoring_system param
  - fallback profile updated to be generic engineering (not Sam-specific)

PUBLIC API (crew.py needs ZERO changes)
────────────────────────────────────────
  score_jobs_with_llm(jobs, api_key, batch_size=20) -> list
  CANDIDATE_PROFILE_FULL  (str)

PIPELINE
────────
Layer 1   BGE-small-en-v1.5  (bi-encoder, cosine similarity)
          >= 0.72  → AUTO-MATCH   (relevance_score=85, priority=HIGH)
          0.50–0.71 → AMBIGUOUS   → Layer 1.5
          <  0.50  → AUTO-REJECT

Layer 1.5 cross-encoder/ms-marco-MiniLM-L-6-v2  (cross-encoder, local CPU)
          Auto-calibrated p25 threshold.
          Above p25 → NIM Layer 2. Below → rejected.

Layer 2   NIM llama-3.3-70b-instruct  (LLM-as-judge, batch scoring)
          Scores cross-encoder survivors. Outputs HIGH/MEDIUM/LOW/SKIP.
          Uses profile-driven scoring axes — no hardcoded role weights.

Layer 3   NIM llama-3.3-70b-instruct  (strict verifier, EDGE CASES ONLY)
          Runs only on HIGH jobs where relevance_score is 65–75.
          Jobs scored >= 76 are trusted — skip verifier entirely.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

try:
    import huggingface_hub.constants as _hf_const
    _hf_const.HF_HUB_OFFLINE = True
except Exception:
    pass

import logging

log = logging.getLogger(__name__)
log.info("scorer imported")

import json
import logging
import time
from pathlib import Path

log.info("json imported , logging,time,pathlab")

import numpy as np
log.info("numpy imported")
log.info("Before sentence_transformers import")
from sentence_transformers import SentenceTransformer, CrossEncoder
log.info("sentence_transformers imported successfully")
log.info("sentence_transformers imported")

from nim_client import make_client, call_nim, clean_json, NIM_SCORING_MODEL
log.info("nim_client imported")

log = logging.getLogger(__name__)

# ── Layer 1 thresholds (BGE bi-encoder) ──────────────────────────────────────
BGE_AUTO_MATCH_THRESHOLD = 0.72
BGE_AUTO_REJECT_THRESHOLD = 0.50

# ── Layer 1.5 auto-calibration percentile ────────────────────────────────────
CE_REJECT_PERCENTILE = 25

# ── Layer 3 edge-case range ───────────────────────────────────────────────────
EDGE_CASE_HIGH_MIN = 65
EDGE_CASE_HIGH_MAX = 75

# ── Model names ───────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# NIM scoring model imported from nim_client: NIM_SCORING_MODEL

# ── Scoring config ────────────────────────────────────────────────────────────
MAX_ATTEMPTS = 2
COVERAGE_MIN = 0.50
INTER_BATCH_SLEEP = 5

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
_PROFILE_PATH = _BASE / "config" / "profile.json"
_UNSCORED_LOG_PATH = _BASE / "logs" / "unscored_jobs.json"


# ── Profile loading ───────────────────────────────────────────────────────────
def _load_profiles() -> tuple[str, str]:
    if _PROFILE_PATH.exists():
        try:
            data = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
            compressed = data.get("_scoring_prompt", "")
            full = data.get("_full_profile", "")
            if compressed:
                log.info(
                    "Profile loaded (~%d tokens compressed, ~%d tokens full)",
                    len(compressed.split()),
                    len(full.split()),
                )
                return compressed, full
        except Exception as e:
            log.warning("Could not load profile.json: %s — using fallback", e)

    log.info(
        "Using hardcoded fallback profile. Run setup_profile.py to use your resume."
    )
    fallback = (
        "Fresher, CSE 2027 batch, India.\n"
        "Skills: Python, Java, JavaScript, React, Node.js, Spring Boot, FastAPI, Docker.\n"
        "Wants: SDE/Backend/Full Stack at product company or startup. 0-2yr exp. India.\n"
        "Veto: TCS/Infosys/Wipro/Accenture/HCL (no engineering title), 3+yr req, non-technical roles."
    )
    return fallback, fallback


CANDIDATE_PROFILE, CANDIDATE_PROFILE_FULL = _load_profiles()


# =============================================================================
# Profile-driven prompt builders  (v2.2 — replaces hardcoded AI/ML prompts)
# =============================================================================

def _build_scoring_system(profile_prompt: str) -> str:
    """
    Build a scoring system prompt from the candidate's actual profile.
    No hardcoded role weights — axes are derived from what THIS candidate wants.
    """
    return f"""Score jobs AND internships for this specific candidate. INTENT matching only, not keywords.

CANDIDATE PROFILE:
{profile_prompt}

SCORING AXES (100pts total):

role(30): How well does the job title and responsibilities match THIS candidate's TARGET ROLES?
  - Exact match to a target role listed above = 28-30
  - Closely related role using same tech stack = 18-27
  - Tangentially related role = 8-17
  - Unrelated to candidate's stack or goals = 0-7

skills(30): How much overlap between job requirements and THIS candidate's TECHNICAL SKILLS?
  - Strong overlap with primary languages/frameworks = 25-30
  - Moderate overlap = 12-24
  - Minimal overlap = 4-11
  - No overlap = 0-3

level(25): Is this role open to THIS candidate's experience level?
  - intern/fresher/entry/0-1yr/campus = 25
  - 0-2yr experience = 20
  - 0-3yr experience = 12
  - 3+yr required = 0

company(15): Company quality and fit for THIS candidate's goals?
  - Top product company / reputable AI or tech startup = 13-15
  - Mid-tier product company / growing startup = 8-12
  - IT services / outsourcing / body-shopping = 0-3

PRIORITY: 85+=HIGH  65-84=MEDIUM  50-64=LOW  <50=SKIP

VETO to SKIP (auto-disqualify without scoring):
  - Any job matching a HARD VETO listed in the candidate profile above
  - Any job type listed in ROLE TYPE EXCLUSIONS in the candidate profile above
  - Roles requiring experience beyond the candidate's max_experience_years
  - Vague/generic titles with no identifiable tech stack in the description
  - Clearly non-engineering roles (sales, marketing, content, HR) unless candidate profile shows that

Be conservative: if description is too thin to confirm a real engineering role, SKIP rather than guess HIGH.
Internships at good companies are equally valid as full-time roles — score them the same way.

CRITICAL: You MUST return a score for EVERY job_id in the input array.
Do not skip or omit any. Return ONLY a valid JSON array, no markdown:
[{{"job_id":"...","relevance_score":0-100,"score_breakdown":{{"role":0,"skills":0,"level":0,"company":0}},"match_reason":"1 sentence","key_match_skills":["s1"],"red_flags":[],"priority":"HIGH|MEDIUM|LOW|SKIP"}}]"""


def _build_verifier_system(profile_prompt: str) -> str:
    """
    Build a verifier prompt from the candidate's actual profile.
    No hardcoded AI/ML references.
    """
    return f"""You are a strict senior recruiter reviewing borderline HIGH job listings.
These were scored HIGH (65-75) by a fast model — your job is to confirm or downgrade.

CANDIDATE PROFILE:
{profile_prompt}

For each listing answer ALL THREE questions:
1. Does this GENUINELY involve hands-on engineering work matching the candidate's primary tech stack
   (not keyword stuffing, not vague "tech" roles)?
2. Is the company a product company, reputable startup, or good tech employer
   (not IT services/outsourcing/body-shopping)?
3. Is this open to the candidate's experience level (fresher/intern/0-2yr/campus hire)?

ALL THREE yes → keep HIGH.
Any doubt → downgrade to MEDIUM.
Clearly wrong, off-profile, or matches a hard veto → SKIP.

Return ONLY valid JSON array:
[{{"job_id":"...","verified_priority":"HIGH|MEDIUM|LOW|SKIP","confidence":0-100,"reason":"one sentence"}}]"""


def _build_profile_anchor(profile_prompt: str) -> str:
    """
    Build BGE embedding anchor from candidate's actual profile.
    Extracts key terms from the scoring prompt instead of hardcoding AI/ML terms.
    Falls back to generic engineering terms if profile is empty.
    """
    if not profile_prompt or len(profile_prompt.strip()) < 50:
        log.warning("Profile prompt too short for anchor — using generic engineering fallback")
        return (
            "Software Engineer Backend Engineer Full Stack Developer "
            "Java Spring Boot Python React Node.js JavaScript "
            "fresher intern entry-level 2027 batch 0-2 years experience India"
        )

    # Extract skills and roles lines from the scoring prompt
    anchor_parts = []
    capture = False
    for line in profile_prompt.split('\n'):
        stripped = line.strip()
        # Capture lines from TECHNICAL SKILLS section
        if 'TECHNICAL SKILLS' in stripped:
            capture = True
            continue
        if capture and stripped and not stripped.startswith('COMPETITIVE'):
            anchor_parts.append(stripped)
        if 'COMPETITIVE' in stripped or 'KEY PROJECTS' in stripped:
            capture = False

        # Always include TARGET ROLES line
        if 'TARGET ROLES' in stripped:
            roles_part = stripped.replace('TARGET ROLES (priority order, derived from resume evidence):', '').strip()
            anchor_parts.append(roles_part)

    # Build anchor — cap at 300 chars to keep embedding focused
    anchor = " ".join(anchor_parts)[:300]

    # Always append level/batch terms
    anchor += " fresher intern entry-level 2027 batch 0-2 years experience India"

    log.info("BGE anchor built (%d chars) from profile", len(anchor))
    return anchor


def _build_ce_query(profile_prompt: str) -> str:
    """
    Build cross-encoder query from candidate's actual profile.
    Extracts target roles and primary stack instead of hardcoding AI/ML.
    """
    if not profile_prompt or len(profile_prompt.strip()) < 50:
        return (
            "Software engineering internship or fresher full-time role "
            "involving programming, backend, or full-stack development. "
            "Open to 2027 batch graduates or interns at product companies or startups."
        )

    roles_line = ""
    skills_line = ""
    batch_line = "2027"

    for line in profile_prompt.split('\n'):
        stripped = line.strip()
        if 'TARGET ROLES' in stripped:
            roles_line = stripped.replace(
                'TARGET ROLES (priority order, derived from resume evidence):', ''
            ).strip()
        if 'TECHNICAL SKILLS' in stripped:
            # grab the next non-empty line as skills summary
            skills_line = stripped
        if 'GRADUATION' in stripped:
            batch_line = stripped

    roles_str = roles_line or "Software Engineer, Backend Engineer, Full Stack Developer"

    return (
        f"Internship or fresher full-time software engineering role. "
        f"Target roles: {roles_str}. "
        f"Open to {batch_line} graduates or interns. "
        f"At a product company or tech startup, not IT outsourcing."
    )


# ── Retry suffix for JSON coverage failures ───────────────────────────────────
_JSON_RETRY_SUFFIX = (
    "\n\nCRITICAL: Your previous response was missing job_ids. "
    "You MUST include a score object for EVERY job_id listed in the input. "
    "Return ONLY a raw JSON array starting with [ and ending with ]. "
    "No markdown, no commentary, no omissions."
)


# ── NIM call helper ───────────────────────────────────────────────────────────
def _nim_call(api_key: str, system_prompt: str, user_content: str) -> str:
    """Create NIM client and call llama-3.3-70b. Returns cleaned JSON string."""
    client = make_client(api_key)
    raw = call_nim(client, system_prompt, user_content, model=NIM_SCORING_MODEL)
    return clean_json(raw)


# =============================================================================
# Shared helpers
# =============================================================================

def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _log_unscored(jobs: list, reason: str) -> None:
    try:
        _UNSCORED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if _UNSCORED_LOG_PATH.exists():
            existing = json.loads(_UNSCORED_LOG_PATH.read_text(encoding="utf-8"))
        for j in jobs:
            existing.append(
                {
                    "job_id": j.get("job_id"),
                    "title": j.get("title"),
                    "company": j.get("company"),
                    "url": j.get("url"),
                    "reason": reason,
                    "date": time.strftime("%Y-%m-%d"),
                }
            )
        _UNSCORED_LOG_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as e:
        log.debug("Could not write unscored audit log: %s", e)


# =============================================================================
# Layer 1 — BGE bi-encoder pre-filter
# =============================================================================

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    print("Entered _get_embed_model")
    if _embed_model is None:
        print("Loading SentenceTransformer...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("SentenceTransformer Loaded")
    return _embed_model


def _build_job_text(job: dict) -> str:
    title = job.get("title", "")
    company = job.get("company", "")
    description = (job.get("description") or job.get("snippet", ""))[:600]
    location = job.get("location", "")
    tags = " ".join(job.get("tags", []) or [])
    return f"{title} {title} at {company}. {location}. {tags}. {description}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _layer1_bge_filter(jobs: list) -> tuple[list, list, list]:
    """Returns (auto_matched, ambiguous, auto_rejected)."""
    if not jobs:
        return [], [], []

    model = _get_embed_model()

    # v2.2: anchor built from actual profile, not hardcoded AI/ML terms
    anchor_text = _build_profile_anchor(CANDIDATE_PROFILE)
    anchor_vec = model.encode(anchor_text, normalize_embeddings=True)

    job_texts = [_build_job_text(j) for j in jobs]
    job_vecs = model.encode(
        job_texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    log.info("BGE encode complete — %d vecs, shape=%s", len(job_vecs), job_vecs.shape)

    auto_matched: list = []
    ambiguous: list = []
    auto_rejected: list = []

    log.info("Starting cosine similarity loop over %d jobs", len(jobs))
    for job, vec in zip(jobs, job_vecs):
        sim = _cosine(anchor_vec, vec)
        job["embedding_sim"] = round(sim, 4)

        if sim >= BGE_AUTO_MATCH_THRESHOLD:
            job.update(
                {
                    "relevance_score": 85,
                    "priority": "HIGH",
                    "score_breakdown": {},
                    "match_reason": f"BGE auto-match (sim={sim:.3f})",
                    "key_match_skills": [],
                    "red_flags": [],
                    "score_source": "embedding_match",
                }
            )
            auto_matched.append(job)
        elif sim >= BGE_AUTO_REJECT_THRESHOLD:
            ambiguous.append(job)
        else:
            job.update(
                {
                    "relevance_score": 0,
                    "priority": "SKIP",
                    "score_source": "embedding_reject",
                }
            )
            auto_rejected.append(job)

    log.info(
        "Layer 1 BGE: %d auto-match | %d ambiguous -> L1.5 | %d auto-reject",
        len(auto_matched),
        len(ambiguous),
        len(auto_rejected),
    )
    return auto_matched, ambiguous, auto_rejected


# =============================================================================
# Layer 1.5 — Cross-encoder re-ranker
# =============================================================================

_ce_model: CrossEncoder | None = None


def _get_ce_model() -> CrossEncoder:
    global _ce_model
    if _ce_model is None:
        log.info(
            "Loading cross-encoder: %s (first run — cached after this)", CE_MODEL_NAME
        )
        _ce_model = CrossEncoder(CE_MODEL_NAME)
    return _ce_model


def _layer1_5_cross_encoder(ambiguous: list) -> tuple[list, list]:
    """Auto-calibrated re-ranking."""
    if not ambiguous:
        return [], []

    model = _get_ce_model()

    # v2.2: query built from actual profile, not hardcoded AI/ML terms
    query = _build_ce_query(CANDIDATE_PROFILE)
    log.info("CE query: %s", query[:120])

    pairs = [(query, _build_job_text(j)) for j in ambiguous]
    scores = model.predict(pairs, show_progress_bar=False)

    for job, score in zip(ambiguous, scores):
        job["ce_score"] = round(float(score), 4)

    score_arr = np.array([float(s) for s in scores])
    reject_thresh = float(np.percentile(score_arr, CE_REJECT_PERCENTILE))

    log.info(
        "Layer 1.5 CE auto-calibrated: reject_thresh=%.3f (p%d) | "
        "score range [%.3f, %.3f]",
        reject_thresh,
        CE_REJECT_PERCENTILE,
        score_arr.min(),
        score_arr.max(),
    )

    to_nim: list = []
    ce_rejected: list = []

    for job in ambiguous:
        if job["ce_score"] >= reject_thresh:
            to_nim.append(job)
        else:
            job.update(
                {
                    "relevance_score": 0,
                    "priority": "SKIP",
                    "score_source": "ce_reject",
                }
            )
            ce_rejected.append(job)

    log.info(
        "Layer 1.5 CE: %d -> NIM | %d auto-reject",
        len(to_nim),
        len(ce_rejected),
    )
    return to_nim, ce_rejected


# =============================================================================
# Layer 2 — NIM scorer
# =============================================================================

def _score_batch_with_retries(
    api_key: str,
    batch: list,
    batch_num: int,
    scoring_system: str,
) -> tuple[list | None, bool]:
    """
    Scores a batch using NIM_SCORING_MODEL with retries.
    scoring_system is now profile-driven — passed in from score_jobs_with_llm().
    """
    payload = [
        {
            "id": j["job_id"],
            "title": j["title"],
            "company": j["company"],
            "loc": (j.get("location") or "")[:30],
            "desc": (j.get("description") or "")[:400],
        }
        for j in batch
    ]
    batch_ids = {j["job_id"] for j in batch}
    last_err = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        force_retry = attempt > 1
        user_content = (
            f"JOBS ({len(payload)} total — score ALL of them): {json.dumps(payload)}\n\n"
            "Return a JSON array with exactly one object per job_id above."
        )
        if force_retry:
            user_content += _JSON_RETRY_SUFFIX

        try:
            raw = _nim_call(api_key, scoring_system, user_content)
            scores = json.loads(raw)

            returned_ids = {s.get("job_id", s.get("id", "")) for s in scores}
            covered = len(returned_ids & batch_ids)
            coverage = covered / max(len(batch_ids), 1)

            if coverage < COVERAGE_MIN:
                last_err = ValueError(
                    f"Low coverage: {covered}/{len(batch_ids)} ({coverage:.0%})"
                )
                log.warning(
                    "  Batch %d attempt %d/%d: %s — retrying",
                    batch_num, attempt, MAX_ATTEMPTS, last_err,
                )
                continue

            if attempt > 1:
                log.info("  Batch %d: recovered on attempt %d", batch_num, attempt)
            return scores, True

        except json.JSONDecodeError as e:
            last_err = e
            log.warning(
                "  Batch %d attempt %d/%d: bad JSON (%s) — retrying",
                batch_num, attempt, MAX_ATTEMPTS, e,
            )

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                wait = 60 if attempt == 1 else 30
                log.info(
                    "  Batch %d attempt %d/%d: rate limited — waiting %ds",
                    batch_num, attempt, MAX_ATTEMPTS, wait,
                )
                time.sleep(wait)
            elif "timeout" in msg or "deadline" in msg:
                log.warning(
                    "  Batch %d attempt %d/%d: timeout — retrying",
                    batch_num, attempt, MAX_ATTEMPTS,
                )
                time.sleep(5)
            else:
                log.error(
                    "  Batch %d attempt %d/%d: %s",
                    batch_num, attempt, MAX_ATTEMPTS, e,
                )
                time.sleep(3)

    log.error("  Batch %d: all attempts failed — last error: %s", batch_num, last_err)
    return None, False


def _layer2_nim_score(
    jobs: list,
    api_key: str,
    batch_size: int,
    scoring_system: str,
) -> tuple[list, int]:
    """Returns (scored_non_skip, unscored_count). score_source='nim_llama'."""
    if not jobs:
        return [], 0

    nim_scored: list = []
    unscored_count: int = 0
    total_batches = -(-len(jobs) // batch_size)

    log.info(
        "Layer 2 NIM: scoring %d jobs | %d batches | model=%s",
        len(jobs), total_batches, NIM_SCORING_MODEL,
    )

    for i in range(0, len(jobs), batch_size):
        batch = jobs[i: i + batch_size]
        batch_num = i // batch_size + 1
        log.info("  Batch %d/%d (%d jobs)...", batch_num, total_batches, len(batch))

        scores, succeeded = _score_batch_with_retries(
            api_key, batch, batch_num, scoring_system
        )

        if not succeeded or scores is None:
            _log_unscored(batch, "all_attempts_failed")
            unscored_count += len(batch)
            log.warning(
                "  Batch %d: excluded %d jobs — see logs/unscored_jobs.json",
                batch_num, len(batch),
            )
            if batch_num < total_batches:
                time.sleep(INTER_BATCH_SLEEP)
            continue

        score_map = {s.get("job_id", s.get("id", "")): s for s in scores}
        passed = skipped = 0

        for job in batch:
            s = score_map.get(job["job_id"])
            if not s:
                _log_unscored([job], "missing_from_nim_response")
                unscored_count += 1
                continue
            if s.get("priority", "SKIP") == "SKIP":
                skipped += 1
                continue
            job.update(
                {
                    "relevance_score": _safe_int(s.get("relevance_score", 0)),
                    "score_breakdown": s.get("score_breakdown", {}),
                    "match_reason": s.get("match_reason", ""),
                    "key_match_skills": s.get("key_match_skills", []),
                    "red_flags": s.get("red_flags", []),
                    "priority": s.get("priority", "LOW"),
                    "score_source": "nim_llama",
                }
            )
            nim_scored.append(job)
            passed += 1

        log.info("    -> %d relevant, %d skipped", passed, skipped)

        if batch_num < total_batches:
            time.sleep(INTER_BATCH_SLEEP)

    return nim_scored, unscored_count


# =============================================================================
# Layer 3 — NIM verifier (edge cases only: HIGH with score 65–75)
# =============================================================================

def _layer3_verify_edge_cases(all_scored: list, api_key: str) -> list:
    """
    Splits scored jobs:
      confirmed_high : relevance_score >= 76  → trusted, skip verifier
      edge_cases     : HIGH with score 65–75  → send to NIM
      non_high       : MEDIUM / LOW           → pass through
    """
    confirmed_high: list = []
    edge_cases: list = []
    non_high: list = []

    for job in all_scored:
        priority = job.get("priority", "LOW")
        score = job.get("relevance_score", 0)

        if priority == "HIGH" and score > EDGE_CASE_HIGH_MAX:
            job["score_source"] = job.get("score_source", "") + "+trusted"
            confirmed_high.append(job)
        elif priority == "HIGH" and EDGE_CASE_HIGH_MIN <= score <= EDGE_CASE_HIGH_MAX:
            edge_cases.append(job)
        else:
            non_high.append(job)

    if not edge_cases:
        log.info(
            "Layer 3: 0 edge-case HIGHs (65-75) — NIM verifier skipped. "
            "%d trusted HIGH, %d MEDIUM/LOW pass-through.",
            len(confirmed_high), len(non_high),
        )
        return confirmed_high + non_high

    log.info(
        "Layer 3 NIM verifier: checking %d edge-case HIGHs (65-75) | "
        "%d trusted HIGHs skip verifier",
        len(edge_cases), len(confirmed_high),
    )

    # v2.2: verifier prompt is also profile-driven
    verifier_system = _build_verifier_system(CANDIDATE_PROFILE)

    try:
        payload = [
            {
                "job_id": j["job_id"],
                "title": j["title"],
                "company": j["company"],
                "job_type": j.get("job_type", "full-time"),
                "score": j.get("relevance_score", 0),
                "reason": (j.get("match_reason") or "")[:120],
            }
            for j in edge_cases
        ]

        user_content = (
            f"Verify these {len(edge_cases)} borderline HIGH jobs.\n"
            f"Return a JSON array with exactly {len(edge_cases)} objects.\n"
            f"{json.dumps(payload)}\nJSON only."
        )
        raw = _nim_call(api_key, verifier_system, user_content)
        results = json.loads(raw)

        result_map = {r["job_id"]: r for r in results}
        downgraded = 0

        for job in edge_cases:
            v = result_map.get(job["job_id"])
            if not v:
                job["score_source"] += "+confirmed_nim_no_verdict"
                confirmed_high.append(job)
                continue

            new_p = v.get("verified_priority", "HIGH")
            conf = v.get("confidence", 100)

            if new_p != "HIGH" and conf >= 85:
                job["priority"] = new_p
                job["match_reason"] = (
                    job.get("match_reason") or ""
                ) + f" [NIM-L3: {v.get('reason', '')}]"
                job["score_source"] += "+downgraded_nim"
                downgraded += 1
                log.info(
                    "  Downgraded: %s @ %s -> %s (conf=%d%%)",
                    job["title"], job["company"], new_p, conf,
                )
                non_high.append(job)
            else:
                job["score_source"] += "+confirmed_nim"
                confirmed_high.append(job)
                log.info(
                    "  Confirmed HIGH: %s @ %s (conf=%d%%)",
                    job["title"], job["company"], conf,
                )

        log.info(
            "Layer 3 done: %d edge-cases checked | %d downgraded | "
            "%d total confirmed HIGH",
            len(edge_cases), downgraded, len(confirmed_high),
        )

    except Exception as e:
        log.error("Layer 3 NIM error: %s — keeping all edge-case HIGHs unchanged", e)
        confirmed_high.extend(edge_cases)

    return confirmed_high + non_high


# =============================================================================
# Public entry point — identical signature to old scorer.py
# =============================================================================

def score_jobs_with_llm(
    jobs: list,
    api_key: str,
    batch_size: int = 20,
) -> list:
    """
    4-layer hybrid scorer. crew.py needs ZERO changes — same signature.

    Layer 1   BGE bi-encoder       → auto-match / ambiguous / auto-reject
    Layer 1.5 Cross-encoder        → re-ranks ambiguous, auto-calibrated p25 threshold
    Layer 2   NIM llama-3.3-70b    → intent-scores ambiguous survivors (profile-driven)
    Layer 3   NIM llama-3.3-70b    → verifies edge-case HIGHs (score 65-75) only

    v2.2: all prompts and anchors are derived from the loaded candidate profile.
    No hardcoded AI/ML role weights — works correctly for any engineering profile.
    """
    if not jobs:
        return []
    if not api_key:
        log.error("nvidia_nim_api_key missing — Layers 2+3 will be skipped.")

    # Layer 1 — BGE (anchor now profile-driven)
    auto_matched, ambiguous, _ = _layer1_bge_filter(jobs)

    # Layer 1.5 — Cross-encoder (query now profile-driven)
    to_nim, _ = _layer1_5_cross_encoder(ambiguous)

    # Layer 2 — NIM (scoring system now profile-driven)
    nim_scored: list = []
    unscored_count: int = 0

    if to_nim and api_key:
        scoring_system = _build_scoring_system(CANDIDATE_PROFILE)
        nim_scored, unscored_count = _layer2_nim_score(
            to_nim, api_key, batch_size, scoring_system
        )
    elif to_nim and not api_key:
        _log_unscored(to_nim, "no_nvidia_nim_api_key")
        unscored_count = len(to_nim)

    # Layer 3 — NIM edge-case verifier (verifier prompt now profile-driven)
    all_scored = auto_matched + nim_scored

    has_edge_cases = any(
        j.get("priority") == "HIGH"
        and EDGE_CASE_HIGH_MIN <= j.get("relevance_score", 0) <= EDGE_CASE_HIGH_MAX
        for j in all_scored
    )

    if api_key and has_edge_cases:
        all_scored = _layer3_verify_edge_cases(all_scored, api_key)
    else:
        log.info("Layer 3: no edge-case HIGHs (65-75) found — NIM verifier skipped.")

    # Final filter — exclude SKIP
    results = [j for j in all_scored if j.get("priority") not in ("SKIP", None)]

    bge_rejected = len(jobs) - len(auto_matched) - len(ambiguous)
    ce_rejected = len(ambiguous) - len(to_nim)

    log.info(
        "Scoring complete — %d relevant from %d jobs "
        "(L1-auto=%d | L1-bge-reject=%d | L1.5-ce-reject=%d | "
        "L2-nim-band=%d | L2-scored=%d | unscored->logs=%d)",
        len(results), len(jobs),
        len(auto_matched), bge_rejected, ce_rejected,
        len(to_nim), len(nim_scored), unscored_count,
    )
    return results