"""
scorer.py — 4-layer hybrid scoring pipeline.

MIGRATION NOTE (v2.0 → v2.1)
──────────────────────────────
Layers 1 (BGE) and 1.5 (cross-encoder) are UNCHANGED — local models, no API.
Layers 2 and 3 previously used Groq; now use Google AI Studio (Gemini).

Why the switch is clean:
  - Layers 2+3 only send text in, get JSON out.
  - Gemini's response_mime_type="application/json" eliminates the ```json
    stripping hack that _clean() was doing.
  - Rate limit: 15 RPM / 1500 RPD / 1M TPM on free tier.
    A typical run (~5 batches of 20) uses ~5 RPM — well within limit.

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
          Above p25 → Gemini Flash. Below → rejected.

Layer 2   Gemini 2.0 Flash  (LLM-as-judge, batch scoring)
          Scores cross-encoder survivors. Outputs HIGH/MEDIUM/LOW/SKIP.
          Fallback: gemini-1.5-flash if flash-2.0 fails.

Layer 3   Gemini 2.0 Flash  (strict verifier, EDGE CASES ONLY)
          Runs only on HIGH jobs where relevance_score is 65–75.
          Jobs scored >= 76 are trusted — skip verifier entirely.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder

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
GEMINI_PRIMARY = "gemini-2.0-flash"
GEMINI_FALLBACK = "gemini-1.5-flash"

# ── Gemini config ─────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 30.0
MAX_ATTEMPTS = 2
COVERAGE_MIN = 0.50
INTER_BATCH_SLEEP = 5  # Gemini free tier: 15 RPM → 4s gap is enough; 5s is safe

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
        "Sam, YCCE Nagpur CSE 2027 fresher, CGPA 7.9, GenAI intern.\n"
        "Skills: LangChain RAG ChromaDB XGBoost FastAPI Python Java AWS Docker CrewAI.\n"
        "Projects: Financial RAG, Credit Risk ML, WebSocket backend SIH, Job Intel Agent.\n"
        "Wants: GenAI/ML/SDE/Backend at product co or AI startup. 0-2yr exp. India.\n"
        "Veto: TCS/Infosys/Wipro/Accenture/HCL(no AI title), 3+yr req, DevOps/Sales/Hardware."
    )
    return fallback, fallback


CANDIDATE_PROFILE, CANDIDATE_PROFILE_FULL = _load_profiles()


# ── Gemini system prompts (unchanged from Groq versions) ─────────────────────
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

_VERIFIER_SYSTEM = """You are a strict senior recruiter reviewing borderline HIGH listings.
These were scored HIGH (65-75) by a fast model — your job is to confirm or downgrade.

For each listing answer:
1. Does this GENUINELY involve AI/ML/backend/software engineering (not keyword stuffing)?
2. Is the company a product company, AI startup, or reputable tech employer (not IT services)?
3. Is this open to a 2027 batch fresher or intern (0-2 years, campus, or explicit intern role)?

ALL THREE yes -> keep HIGH.
Any doubt -> downgrade to MEDIUM.
Clearly wrong, scam-like, or off-profile -> SKIP.

Return ONLY valid JSON array:
[{"job_id":"...","verified_priority":"HIGH|MEDIUM|LOW|SKIP","confidence":0-100,"reason":"one sentence"}]"""

_JSON_RETRY_SUFFIX = (
    "\n\nCRITICAL: Your previous response was missing job_ids. "
    "You MUST include a score object for EVERY job_id listed in the input. "
    "Return ONLY a raw JSON array starting with [ and ending with ]. "
    "No markdown, no commentary, no omissions."
)


# ── Gemini client factory ─────────────────────────────────────────────────────


def _make_gemini_model(api_key: str, model_name: str) -> genai.GenerativeModel:
    """Configure genai globally and return a model instance."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json",  # forces valid JSON — no ```json stripping needed
        ),
    )


def _call_gemini(
    model: genai.GenerativeModel,
    system_prompt: str,
    user_content: str,
) -> str:
    """
    Single Gemini call. Returns raw response text.
    Gemini doesn't have a native system/user split in the SDK the same way
    as OpenAI — we prepend the system prompt to the user message.
    response_mime_type="application/json" handles JSON enforcement.
    """
    full_prompt = f"{system_prompt}\n\n{user_content}"
    response = model.generate_content(full_prompt)
    return response.text


# =============================================================================
# Shared helpers (unchanged from original)
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
# Layer 1 — BGE bi-encoder pre-filter  (UNCHANGED)
# =============================================================================

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info("Loading bi-encoder: %s", EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _build_profile_anchor(_: str) -> str:
    return (
        "Generative AI Engineer LLM Engineer Machine Learning Engineer "
        "Software Engineer Backend Engineer Full Stack Developer Data Scientist "
        "Python LangChain LangGraph CrewAI RAG LLM AWS Bedrock FastAPI Docker "
        "React Node.js XGBoost ChromaDB Vector Embeddings "
        "fresher intern entry-level 2027 batch 0-2 years experience"
    )


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
    """Returns (auto_matched, ambiguous, auto_rejected). UNCHANGED."""
    if not jobs:
        return [], [], []

    model = _get_embed_model()
    anchor_vec = model.encode(
        _build_profile_anchor(CANDIDATE_PROFILE),
        normalize_embeddings=True,
    )
    job_texts = [_build_job_text(j) for j in jobs]
    job_vecs = model.encode(
        job_texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )

    auto_matched: list = []
    ambiguous: list = []
    auto_rejected: list = []

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
# Layer 1.5 — Cross-encoder re-ranker  (UNCHANGED)
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


def _build_ce_query() -> str:
    return (
        "Software engineering or AI/ML internship or fresher full-time role "
        "involving Python, LLMs, RAG, LangChain, machine learning, backend, "
        "or full-stack development. Open to 2027 batch graduates or interns. "
        "At a product company or AI startup, not IT outsourcing."
    )


def _layer1_5_cross_encoder(ambiguous: list) -> tuple[list, list]:
    """Auto-calibrated re-ranking. UNCHANGED."""
    if not ambiguous:
        return [], []

    model = _get_ce_model()
    query = _build_ce_query()
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

    to_groq: list = []
    ce_rejected: list = []

    for job in ambiguous:
        if job["ce_score"] >= reject_thresh:
            to_groq.append(job)
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
        "Layer 1.5 CE: %d -> Gemini | %d auto-reject",
        len(to_groq),
        len(ce_rejected),
    )
    return to_groq, ce_rejected


# =============================================================================
# Layer 2 — Gemini scorer  (replaces Groq 8B)
# =============================================================================


def _score_batch_with_retries(
    api_key: str,
    batch: list,
    batch_num: int,
) -> tuple[list | None, bool]:
    """
    Tries GEMINI_PRIMARY first, falls back to GEMINI_FALLBACK on failure.
    Mirrors the old _score_batch_with_retries behaviour exactly.
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

    models_to_try = [GEMINI_PRIMARY, GEMINI_FALLBACK]

    for attempt, model_name in enumerate(models_to_try, start=1):
        force_retry = attempt > 1
        user_content = (
            f"CANDIDATE: {CANDIDATE_PROFILE}\n\n"
            f"JOBS ({len(payload)} total — score ALL of them): {json.dumps(payload)}\n\n"
            "Return a JSON array with exactly one object per job_id above."
        )
        if force_retry:
            user_content += _JSON_RETRY_SUFFIX

        try:
            model = _make_gemini_model(api_key, model_name)
            raw = _call_gemini(model, SCORING_SYSTEM, user_content)
            scores = json.loads(raw)

            returned_ids = {s.get("job_id", s.get("id", "")) for s in scores}
            covered = len(returned_ids & batch_ids)
            coverage = covered / max(len(batch_ids), 1)

            if coverage < COVERAGE_MIN:
                last_err = ValueError(
                    f"Low coverage: {covered}/{len(batch_ids)} ({coverage:.0%})"
                )
                log.warning(
                    "  Batch %d attempt %d/%d (%s): %s — retrying",
                    batch_num,
                    attempt,
                    MAX_ATTEMPTS,
                    model_name,
                    last_err,
                )
                continue

            if attempt > 1:
                log.info(
                    "  Batch %d: recovered on attempt %d using %s",
                    batch_num,
                    attempt,
                    model_name,
                )
            return scores, True

        except json.JSONDecodeError as e:
            last_err = e
            log.warning(
                "  Batch %d attempt %d/%d (%s): bad JSON (%s) — retrying",
                batch_num,
                attempt,
                len(models_to_try),
                model_name,
                e,
            )

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                wait = 60 if attempt == 1 else 30
                log.info(
                    "  Batch %d attempt %d/%d (%s): rate limited — waiting %ds",
                    batch_num,
                    attempt,
                    len(models_to_try),
                    model_name,
                    wait,
                )
                time.sleep(wait)
            elif "timeout" in msg or "deadline" in msg:
                log.warning(
                    "  Batch %d attempt %d/%d (%s): timeout — retrying",
                    batch_num,
                    attempt,
                    len(models_to_try),
                    model_name,
                )
                time.sleep(5)
            else:
                log.error(
                    "  Batch %d attempt %d/%d (%s): %s",
                    batch_num,
                    attempt,
                    len(models_to_try),
                    model_name,
                    e,
                )
                time.sleep(3)

    log.error(
        "  Batch %d: all attempts failed — last error: %s",
        batch_num,
        last_err,
    )
    return None, False


def _layer2_gemini_score(
    jobs: list,
    api_key: str,
    batch_size: int,
) -> tuple[list, int]:
    """Returns (scored_non_skip, unscored_count). score_source='gemini_flash'."""
    if not jobs:
        return [], 0

    gemini_scored: list = []
    unscored_count: int = 0
    total_batches = -(-len(jobs) // batch_size)

    log.info(
        "Layer 2 Gemini: scoring %d jobs | %d batches | primary=%s fallback=%s",
        len(jobs),
        total_batches,
        GEMINI_PRIMARY,
        GEMINI_FALLBACK,
    )

    for i in range(0, len(jobs), batch_size):
        batch = jobs[i : i + batch_size]
        batch_num = i // batch_size + 1
        log.info("  Batch %d/%d (%d jobs)...", batch_num, total_batches, len(batch))

        scores, succeeded = _score_batch_with_retries(api_key, batch, batch_num)

        if not succeeded or scores is None:
            _log_unscored(batch, "all_attempts_failed")
            unscored_count += len(batch)
            log.warning(
                "  Batch %d: excluded %d jobs — see logs/unscored_jobs.json",
                batch_num,
                len(batch),
            )
            if batch_num < total_batches:
                time.sleep(INTER_BATCH_SLEEP)
            continue

        score_map = {s.get("job_id", s.get("id", "")): s for s in scores}
        passed = skipped = 0

        for job in batch:
            s = score_map.get(job["job_id"])
            if not s:
                _log_unscored([job], "missing_from_gemini_response")
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
                    "score_source": "gemini_flash",
                }
            )
            gemini_scored.append(job)
            passed += 1

        log.info("    -> %d relevant, %d skipped", passed, skipped)

        if batch_num < total_batches:
            time.sleep(INTER_BATCH_SLEEP)

    return gemini_scored, unscored_count


# =============================================================================
# Layer 3 — Gemini verifier (edge cases only: HIGH with score 65–75)
# replaces Groq 70B
# =============================================================================


def _layer3_verify_edge_cases(all_scored: list, api_key: str) -> list:
    """
    Splits scored jobs:
      confirmed_high : relevance_score >= 76  → trusted, skip verifier
      edge_cases     : HIGH with score 65–75  → send to Gemini
      non_high       : MEDIUM / LOW           → pass through

    CHANGED: takes api_key instead of Groq client.
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
            "Layer 3: 0 edge-case HIGHs (65-75) — Gemini verifier skipped. "
            "%d trusted HIGH, %d MEDIUM/LOW pass-through.",
            len(confirmed_high),
            len(non_high),
        )
        return confirmed_high + non_high

    log.info(
        "Layer 3 Gemini verifier: checking %d edge-case HIGHs (65-75) | "
        "%d trusted HIGHs skip verifier",
        len(edge_cases),
        len(confirmed_high),
    )

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

        model = _make_gemini_model(api_key, GEMINI_PRIMARY)
        user_content = (
            f"Verify these {len(edge_cases)} borderline HIGH jobs.\n"
            f"Return a JSON array with exactly {len(edge_cases)} objects.\n"
            f"{json.dumps(payload)}\nJSON only."
        )
        raw = _call_gemini(model, _VERIFIER_SYSTEM, user_content)
        results = json.loads(raw)

        result_map = {r["job_id"]: r for r in results}
        downgraded = 0

        for job in edge_cases:
            v = result_map.get(job["job_id"])
            if not v:
                job["score_source"] += "+confirmed_gemini_no_verdict"
                confirmed_high.append(job)
                continue

            new_p = v.get("verified_priority", "HIGH")
            conf = v.get("confidence", 100)

            if new_p != "HIGH" and conf >= 85:
                job["priority"] = new_p
                job["match_reason"] = (
                    job.get("match_reason") or ""
                ) + f" [Gemini-L3: {v.get('reason', '')}]"
                job["score_source"] += "+downgraded_gemini"
                downgraded += 1
                log.info(
                    "  Downgraded: %s @ %s -> %s (conf=%d%%)",
                    job["title"],
                    job["company"],
                    new_p,
                    conf,
                )
                non_high.append(job)
            else:
                job["score_source"] += "+confirmed_gemini"
                confirmed_high.append(job)
                log.info(
                    "  Confirmed HIGH: %s @ %s (conf=%d%%)",
                    job["title"],
                    job["company"],
                    conf,
                )

        log.info(
            "Layer 3 done: %d edge-cases checked | %d downgraded | "
            "%d total confirmed HIGH",
            len(edge_cases),
            downgraded,
            len(confirmed_high),
        )

    except Exception as e:
        log.error("Layer 3 Gemini error: %s — keeping all edge-case HIGHs unchanged", e)
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

    Layer 1   BGE bi-encoder  → auto-match / ambiguous / auto-reject
    Layer 1.5 Cross-encoder   → re-ranks ambiguous, auto-calibrated p25 threshold
    Layer 2   Gemini Flash    → intent-scores ambiguous survivors
    Layer 3   Gemini Flash    → verifies edge-case HIGHs (score 65-75) only

    api_key is now a Google AI Studio key (was Groq key).
    """
    if not jobs:
        return []
    if not api_key:
        log.error("google_ai_api_key missing — Layers 2+3 will be skipped.")

    # Layer 1 — BGE
    auto_matched, ambiguous, _ = _layer1_bge_filter(jobs)

    # Layer 1.5 — Cross-encoder
    to_gemini, _ = _layer1_5_cross_encoder(ambiguous)

    # Layer 2 — Gemini
    gemini_scored: list = []
    unscored_count: int = 0

    if to_gemini and api_key:
        gemini_scored, unscored_count = _layer2_gemini_score(
            to_gemini, api_key, batch_size
        )
    elif to_gemini and not api_key:
        _log_unscored(to_gemini, "no_google_ai_api_key")
        unscored_count = len(to_gemini)

    # Layer 3 — Gemini edge-case verifier
    all_scored = auto_matched + gemini_scored

    has_edge_cases = any(
        j.get("priority") == "HIGH"
        and EDGE_CASE_HIGH_MIN <= j.get("relevance_score", 0) <= EDGE_CASE_HIGH_MAX
        for j in all_scored
    )

    if api_key and has_edge_cases:
        all_scored = _layer3_verify_edge_cases(all_scored, api_key)
    else:
        log.info("Layer 3: no edge-case HIGHs (65-75) found — Gemini verifier skipped.")

    # Final filter — exclude SKIP
    results = [j for j in all_scored if j.get("priority") not in ("SKIP", None)]

    bge_rejected = len(jobs) - len(auto_matched) - len(ambiguous)
    ce_rejected = len(ambiguous) - len(to_gemini)

    log.info(
        "Scoring complete — %d relevant from %d jobs "
        "(L1-auto=%d | L1-bge-reject=%d | L1.5-ce-reject=%d | "
        "L2-gemini-band=%d | L2-scored=%d | unscored->logs=%d)",
        len(results),
        len(jobs),
        len(auto_matched),
        bge_rejected,
        ce_rejected,
        len(to_gemini),
        len(gemini_scored),
        unscored_count,
    )
    return results
