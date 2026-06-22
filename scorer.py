"""
scorer.py — Hybrid scoring: local embeddings → Groq (ambiguous band only).

PIPELINE
--------
Stage 1  Local embedding pre-filter (bge-small-en-v1.5)
         Cosine similarity: profile anchor vs job text

         >= 0.85  AUTO-MATCH   relevance_score=85, priority="HIGH",  skip Groq
         0.70-0.84 AMBIGUOUS   → Stage 2 (Groq)
         <  0.70  AUTO-REJECT  relevance_score=0,  priority="SKIP",  skip Groq

Stage 2  Groq LLM scoring (ambiguous band only)
         Primary model : llama-3.1-8b-instant  (20k TPM)
         Fallback model: llama-3.3-70b-versatile
         Batch size 20, inter-batch delay 8s
         Coverage check >= 50% before accepting response
         Failures → logs/unscored_jobs.json (never injected as LOW)

PUBLIC API  (identical to old scorer.py — crew.py needs zero changes)
----------
  score_jobs_with_llm(jobs, api_key, batch_size=20) -> list
  CANDIDATE_PROFILE_FULL  (str)
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
AUTO_MATCH_THRESHOLD  = 0.85   # >= this → skip Groq, accept as HIGH
AUTO_REJECT_THRESHOLD = 0.70   # <  this → skip Groq, reject as SKIP

# ── Groq config (unchanged from old scorer.py) ────────────────────────────────
SCORING_MODEL_PRIMARY   = "llama-3.1-8b-instant"
SCORING_MODEL_SECONDARY = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT         = 30.0
MAX_ATTEMPTS            = 2
COVERAGE_MIN            = 0.50
INTER_BATCH_SLEEP       = 8

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE              = Path(__file__).parent
_PROFILE_PATH      = _BASE / "config" / "profile.json"
_UNSCORED_LOG_PATH = _BASE / "logs" / "unscored_jobs.json"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ── Module-level profile (keeps CANDIDATE_PROFILE_FULL export for crew.py) ───
def _load_profiles() -> tuple[str, str]:
    if _PROFILE_PATH.exists():
        try:
            data       = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
            compressed = data.get("_scoring_prompt", "")
            full       = data.get("_full_profile", "")
            if compressed:
                log.info(
                    "Profile loaded (~%d tokens compressed, ~%d tokens full)",
                    len(compressed.split()), len(full.split()),
                )
                return compressed, full
        except Exception as e:
            log.warning("Could not load profile.json: %s — using fallback", e)

    log.info("Using hardcoded fallback profile. Run setup_profile.py to use your resume.")
    fallback = (
        "Sam, YCCE Nagpur CSE 2027 fresher, CGPA 7.9, GenAI intern.\n"
        "Skills: LangChain RAG ChromaDB XGBoost FastAPI Python Java AWS Docker CrewAI.\n"
        "Projects: Financial RAG, Credit Risk ML, WebSocket backend SIH, Job Intel Agent.\n"
        "Wants: GenAI/ML/SDE/Backend at product co or AI startup. 0-2yr exp. India.\n"
        "Veto: TCS/Infosys/Wipro/Accenture/HCL(no AI title), 3+yr req, DevOps/Sales/Hardware."
    )
    return fallback, fallback


CANDIDATE_PROFILE, CANDIDATE_PROFILE_FULL = _load_profiles()


# ── Groq system prompt (unchanged from old scorer.py) ─────────────────────────
SCORING_SYSTEM = """Score jobs AND internships for this candidate. INTENT matching only, not keywords.

CANDIDATE wants: full-time fresher roles AND paid internships (especially with PPO/stipend).
Both job types are equally valid. Internship at Google/Amazon/top startup = HIGH.

AXES (100pts total):
role(30): GenAI/LLM=30 ML=25 SDE@AI-co=20 SDE=14 DevOps=3
skills(30): Python+LLM/ML=28 Python+backend=20 irrelevant=2
level(25): intern/fresher/entry/0-1yr=25 0-2yr=20 0-3yr=12 3+yr required=0
company(15): top-product/AI-startup=15 mid-product=10 IT-services=2

PRIORITY: 85+=HIGH 65-84=MEDIUM 50-64=LOW <50=SKIP

VETO to SKIP: any job that matches a HARD VETO or ROLE TYPE EXCLUSION listed
in the CANDIDATE section below — these are derived from this specific
candidate's resume. Also SKIP: vague/generic titles with no identifiable
tech stack.

Be conservative: if description is too thin to confirm a real engineering
role, SKIP rather than guess HIGH.

CRITICAL: You MUST return a score for EVERY job_id in the input array.
Do not skip or omit any. Return ONLY a valid JSON array, no markdown:
[{"job_id":"...","relevance_score":0-100,"score_breakdown":{"role":0,"skills":0,"level":0,"company":0},"match_reason":"1 sentence","key_match_skills":["s1"],"red_flags":[],"priority":"HIGH|MEDIUM|LOW|SKIP"}]"""

_JSON_RETRY_SUFFIX = (
    "\n\nCRITICAL: Your previous response was missing job_ids. "
    "You MUST include a score object for EVERY job_id listed in the input. "
    "Return ONLY a raw JSON array starting with [ and ending with ]. "
    "No markdown, no commentary, no omissions."
)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — Local embedding pre-filter
# ═════════════════════════════════════════════════════════════════════════════

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info("Loading embedding model: %s (first run — cached after this)", EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _build_profile_anchor(profile_text: str) -> str:
    """Use the compressed scoring prompt as the anchor — it already encodes
    target roles, skills, and vetoes in dense form."""
    return profile_text


def _build_job_text(job: dict) -> str:
    title       = job.get("title", "")
    company     = job.get("company", "")
    description = (job.get("description") or job.get("snippet", ""))[:400]
    location    = job.get("location", "")
    return f"{title} at {company}. {location}. {description}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _embedding_prefilter(
    jobs: list,
) -> tuple[list, list, list]:
    """
    Returns (auto_matched, ambiguous, auto_rejected).
    auto_matched  jobs get: relevance_score=85, priority="HIGH",  score_source="embedding_match"
    auto_rejected jobs get: relevance_score=0,  priority="SKIP",  score_source="embedding_reject"
    ambiguous     jobs get: embedding_sim set for debug, routed to Groq
    """
    if not jobs:
        return [], [], []

    model       = _get_embed_model()
    anchor_vec  = model.encode(
        _build_profile_anchor(CANDIDATE_PROFILE),
        normalize_embeddings=True,
    )
    job_texts   = [_build_job_text(j) for j in jobs]
    job_vecs    = model.encode(job_texts, normalize_embeddings=True,
                               batch_size=64, show_progress_bar=False)

    auto_matched:  list = []
    ambiguous:     list = []
    auto_rejected: list = []

    for job, vec in zip(jobs, job_vecs):
        sim = _cosine(anchor_vec, vec)
        job["embedding_sim"] = round(sim, 4)

        if sim >= AUTO_MATCH_THRESHOLD:
            job["relevance_score"]  = 85
            job["priority"]         = "HIGH"
            job["score_breakdown"]  = {}
            job["match_reason"]     = f"Strong embedding match (sim={sim:.3f})"
            job["key_match_skills"] = []
            job["red_flags"]        = []
            job["score_source"]     = "embedding_match"
            auto_matched.append(job)

        elif sim >= AUTO_REJECT_THRESHOLD:
            ambiguous.append(job)

        else:
            job["relevance_score"] = 0
            job["priority"]        = "SKIP"
            job["score_source"]    = "embedding_reject"
            auto_rejected.append(job)

    log.info(
        "Embedding pre-filter: %d auto-match | %d → Groq | %d auto-reject",
        len(auto_matched), len(ambiguous), len(auto_rejected),
    )
    return auto_matched, ambiguous, auto_rejected


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — Groq scoring (ambiguous band only, logic unchanged from old file)
# ═════════════════════════════════════════════════════════════════════════════

def _clean(raw: str) -> str:
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:].strip()
    return raw.strip()


def _safe_int(value, default=0) -> int:
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
            existing.append({
                "job_id":  j.get("job_id"),
                "title":   j.get("title"),
                "company": j.get("company"),
                "url":     j.get("url"),
                "reason":  reason,
                "date":    time.strftime("%Y-%m-%d"),
            })
        _UNSCORED_LOG_PATH.write_text(
            json.dumps(existing, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.debug("Could not write unscored audit log: %s", e)


def _call_groq_batch(
    client: Groq,
    payload: list,
    model: str,
    force_json_retry: bool = False,
) -> list:
    """Single Groq call. Returns parsed JSON list or raises."""
    user_content = (
        f"CANDIDATE: {CANDIDATE_PROFILE}\n\n"
        f"JOBS ({len(payload)} total — score ALL of them): {json.dumps(payload)}\n\n"
        "Return a JSON array with exactly one object per job_id above."
    )
    if force_json_retry:
        user_content += _JSON_RETRY_SUFFIX

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SCORING_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.1,
        max_tokens=2800,
        timeout=REQUEST_TIMEOUT,
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(_clean(raw))


def _score_batch_with_retries(
    client: Groq,
    batch: list,
    batch_num: int,
) -> tuple[list | None, bool]:
    """
    Try up to MAX_ATTEMPTS to score a batch.
    Attempt 1: primary model (8b-instant, high TPM).
    Attempt 2: secondary model (70b-versatile) + JSON retry hint.
    Returns (scores, succeeded).
    """
    payload = [
        {
            "id":      j["job_id"],
            "title":   j["title"],
            "company": j["company"],
            "loc":     (j.get("location") or "")[:30],
            "desc":    (j.get("description") or "")[:100],
        }
        for j in batch
    ]
    batch_ids = {j["job_id"] for j in batch}
    last_err  = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        model      = SCORING_MODEL_PRIMARY if attempt == 1 else SCORING_MODEL_SECONDARY
        force_json = attempt > 1

        try:
            scores       = _call_groq_batch(client, payload, model, force_json_retry=force_json)
            returned_ids = {s.get("job_id", s.get("id", "")) for s in scores}
            covered      = len(returned_ids & batch_ids)
            coverage     = covered / max(len(batch_ids), 1)

            if coverage < COVERAGE_MIN:
                last_err = ValueError(
                    f"Low coverage: {covered}/{len(batch_ids)} ({coverage:.0%})"
                )
                log.warning(
                    "  Batch %d attempt %d/%d (%s): %s — retrying",
                    batch_num, attempt, MAX_ATTEMPTS, model, last_err,
                )
                continue

            if attempt > 1:
                log.info("  Batch %d: recovered on attempt %d using %s", batch_num, attempt, model)
            return scores, True

        except json.JSONDecodeError as e:
            last_err = e
            log.warning(
                "  Batch %d attempt %d/%d (%s): bad JSON (%s) — retrying",
                batch_num, attempt, MAX_ATTEMPTS, model, e,
            )

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "rate_limit" in msg or "429" in msg:
                wait = 60 if attempt == 1 else 30
                log.info(
                    "  Batch %d attempt %d/%d (%s): rate limited — waiting %ds",
                    batch_num, attempt, MAX_ATTEMPTS, model, wait,
                )
                time.sleep(wait)
            elif "timeout" in msg:
                log.warning(
                    "  Batch %d attempt %d/%d (%s): timeout — retrying",
                    batch_num, attempt, MAX_ATTEMPTS, model,
                )
                time.sleep(5)
            else:
                log.error(
                    "  Batch %d attempt %d/%d (%s): %s",
                    batch_num, attempt, MAX_ATTEMPTS, model, e,
                )
                time.sleep(3)

    log.error(
        "  Batch %d: all %d attempts failed — last error: %s",
        batch_num, MAX_ATTEMPTS, last_err,
    )
    return None, False


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point — identical signature to old scorer.py
# ═════════════════════════════════════════════════════════════════════════════

def score_jobs_with_llm(
    jobs: list,
    api_key: str,
    batch_size: int = 20,
) -> list:
    """
    Hybrid scorer. Drop-in replacement for the old score_jobs_with_llm.

    Stage 1: embedding pre-filter (bge-small-en-v1.5)
      >= 0.85  → AUTO-MATCH  (HIGH, skip Groq)
      0.70-0.84 → AMBIGUOUS  → Stage 2
      <  0.70  → AUTO-REJECT (SKIP, skip Groq)

    Stage 2: Groq scoring on ambiguous band only
      Same batch logic, retry logic, and output fields as before.

    Returns list of jobs with priority HIGH/MEDIUM/LOW (SKIP excluded),
    same fields as before: relevance_score, priority, score_breakdown,
    match_reason, key_match_skills, red_flags.
    """
    if not jobs:
        return []
    if not api_key:
        log.error("groq_api_key missing — cannot score ambiguous band.")

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    auto_matched, ambiguous, _ = _embedding_prefilter(jobs)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    groq_scored: list = []
    unscored_count    = 0

    if ambiguous and api_key:
        client        = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT)
        total_batches = -(-len(ambiguous) // batch_size)
        log.info(
            "Groq scoring %d ambiguous jobs | %d batches of %d | "
            "primary=%s fallback=%s",
            len(ambiguous), total_batches, batch_size,
            SCORING_MODEL_PRIMARY, SCORING_MODEL_SECONDARY,
        )

        for i in range(0, len(ambiguous), batch_size):
            batch     = ambiguous[i : i + batch_size]
            batch_num = i // batch_size + 1
            log.info("  Batch %d/%d (%d jobs)…", batch_num, total_batches, len(batch))

            scores, succeeded = _score_batch_with_retries(client, batch, batch_num)

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

            score_map  = {s.get("job_id", s.get("id", "")): s for s in scores}
            passed = skipped = 0

            for job in batch:
                s = score_map.get(job["job_id"])
                if not s:
                    _log_unscored([job], "missing_from_groq_response")
                    unscored_count += 1
                    continue
                if s.get("priority", "SKIP") == "SKIP":
                    skipped += 1
                    continue
                job["relevance_score"]  = _safe_int(s.get("relevance_score", 0))
                job["score_breakdown"]  = s.get("score_breakdown", {})
                job["match_reason"]     = s.get("match_reason", "")
                job["key_match_skills"] = s.get("key_match_skills", [])
                job["red_flags"]        = s.get("red_flags", [])
                job["priority"]         = s.get("priority", "LOW")
                job["score_source"]     = "groq"
                groq_scored.append(job)
                passed += 1

            log.info("    -> %d relevant, %d skipped", passed, skipped)

            if batch_num < total_batches:
                time.sleep(INTER_BATCH_SLEEP)

    elif ambiguous and not api_key:
        _log_unscored(ambiguous, "no_groq_api_key")
        unscored_count += len(ambiguous)

    # ── Merge & report ────────────────────────────────────────────────────────
    results = auto_matched + groq_scored

    log.info(
        "Scoring complete — %d relevant from %d jobs "
        "(auto-match=%d groq=%d | %d rejected by embedding | %d unscored → logs)",
        len(results), len(jobs),
        len(auto_matched), len(groq_scored),
        len(jobs) - len(auto_matched) - len(ambiguous),
        unscored_count,
    )
    return results