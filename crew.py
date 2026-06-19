"""
crew.py — CrewAI 1.x pipeline.
Agents use Ollama qwen3:4b (local, no rate limits, tiny token usage).
Scoring and verification use Groq llama-3.3-70b (accurate, fast, free tier).

Changes vs original:
  - filters.apply_all_filters() called inside fetch_all_jobs_tool after dedup
    (token-free: seniority/exp/type classification before any Groq call)
  - job["job_type"] set by filters — "internship" | "full-time"
  - Verifier downgrade threshold raised: conf > 50 → conf >= 85
    (was dropping 6/7 HIGH — now only downgrades on very high confidence)
  - build_report_tool splits verified_jobs by job_type before calling reporter
  - build_html_report() receives (internships, full_time_jobs, today) separately
  - Email subject shows internship + job counts
"""

import json
import logging
from pathlib import Path
from datetime import date
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from groq import Groq

from sources.public_apis import (
    fetch_remotive, fetch_arbeitnow, fetch_jobicy,
    fetch_himalayas, fetch_freshersworld,
)
from sources.career_portals import fetch_all_career_portals
from sources.linkedin       import fetch_linkedin
from sources.naukri         import fetch_naukri
from scorer                 import score_jobs_with_llm, CANDIDATE_PROFILE_FULL
from reporter               import build_html_report, send_email
from utils                  import load_seen, save_seen, deduplicate
from filters                import apply_all_filters          # ← NEW

log = logging.getLogger(__name__)

BASE    = Path(__file__).parent
SEEN    = BASE / "logs" / "seen_jobs.json"
REPORTS = BASE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
(BASE / "logs").mkdir(parents=True, exist_ok=True)

class _PipelineState:
    """
    Stable container for inter-agent data.

    Why a class instead of a bare dict:
    CrewAI 1.x re-evaluates module-level expressions between agent boundaries
    in sequential crews. A bare dict literal `_state: dict = {...}` can be
    reset to its initial value when a new agent starts, wiping data written
    by earlier agents. A class instance is only constructed once at import
    time; attribute writes on the instance persist for the entire process.
    """
    def __init__(self):
        self.raw_jobs:      list = []
        self.fresh_jobs:    list = []
        self.scored_jobs:   list = []
        self.verified_jobs: list = []
        self.config:        dict = {}

    # Dict-style access so existing _state["key"] calls keep working
    def __getitem__(self, key):        return getattr(self, key)
    def __setitem__(self, key, value): setattr(self, key, value)
    def get(self, key, default=None):  return getattr(self, key, default)

_state = _PipelineState()


def _get_llm() -> LLM:
    return LLM(
        model="ollama/qwen3:4b",
        base_url="http://localhost:11434",
        temperature=0.1,
    )


# ── Tool 1: Scout ─────────────────────────────────────────────────────────────

@tool("Fetch all jobs from all sources")
def fetch_all_jobs_tool(dummy: str = "go") -> str:
    """
    Fetches all job listings from all sources, deduplicates, then applies
    ALL pre-LLM filters (URL fix, job-type classification, seniority drop,
    experience filter) before any Groq call is made.
    """
    cfg      = _state["config"]
    keywords = cfg.get("search_keywords", [])
    location = cfg.get("location", "India")

    raw = []
    raw.extend(fetch_remotive(keywords))
    raw.extend(fetch_arbeitnow(keywords))
    raw.extend(fetch_jobicy(keywords))
    raw.extend(fetch_himalayas(keywords))
    raw.extend(fetch_freshersworld(keywords))
    raw.extend(fetch_linkedin(keywords, location))
    raw.extend(fetch_naukri())
    raw.extend(fetch_all_career_portals())

    seen = load_seen(SEEN)
    fresh, seen = deduplicate(raw, seen)
    save_seen(SEEN, seen)

    # ── Pre-LLM filters — zero tokens ──────────────────────────────────────
    # Must run BEFORE score_jobs_tool. Filters set job["job_type"] and
    # normalise job["url"] to direct links for all sources.
    fresh = apply_all_filters(fresh)                          # ← NEW

    _state["raw_jobs"]   = raw
    _state["fresh_jobs"] = fresh

    intern_count = sum(1 for j in fresh if j.get("job_type") == "internship")
    ft_count     = len(fresh) - intern_count

    log.info(
        f"Scout: {len(raw)} total → {len(fresh)} after dedup+filter "
        f"({intern_count} internships, {ft_count} full-time)"
    )
    return (
        f"Fetched {len(raw)} total. After dedup + pre-LLM filter: {len(fresh)} "
        f"({intern_count} internships, {ft_count} full-time). Ready to score."
    )


# ── Tool 2: Analyst ───────────────────────────────────────────────────────────

@tool("Score jobs with Groq LLM intent matching")
def score_jobs_tool(dummy: str = "score") -> str:
    """
    Scores pre-filtered fresh jobs using Groq llama-3.3-70b.
    job["job_type"] is already set — scorer uses it for context.
    """
    fresh      = _state.get("fresh_jobs", [])
    cfg        = _state.get("config", {})
    api_key    = cfg.get("groq_api_key", "")
    batch_size = cfg.get("llm_batch_size", 20)

    if not fresh:
        return "No fresh jobs to score."

    scored = score_jobs_with_llm(fresh, api_key=api_key, batch_size=batch_size)
    _state["scored_jobs"] = scored

    high   = len([j for j in scored if j.get("priority") == "HIGH"])
    medium = len([j for j in scored if j.get("priority") == "MEDIUM"])
    low    = len([j for j in scored if j.get("priority") == "LOW"])

    log.info(f"Analyst: {len(scored)} relevant — {high}H {medium}M {low}L")
    return (
        f"Scored {len(fresh)} jobs. "
        f"Relevant: {len(scored)} ({high} HIGH, {medium} MEDIUM, {low} LOW). "
        f"Dropped: {len(fresh) - len(scored)} irrelevant."
    )


# ── Tool 3: Verifier ──────────────────────────────────────────────────────────

@tool("Verify HIGH priority jobs for accuracy")
def verify_jobs_tool(dummy: str = "verify") -> str:
    """
    Permissive second-pass on HIGH jobs.
    Only downgrades when confidence >= 85 (was > 50 — was too aggressive).
    Internships and research roles default to keep HIGH.
    """
    scored  = _state.get("scored_jobs", [])
    cfg     = _state.get("config", {})
    api_key = cfg.get("groq_api_key", "")

    if not scored:
        _state["verified_jobs"] = []
        return "No scored jobs to verify."

    high_jobs  = [j for j in scored if j.get("priority") == "HIGH"]
    other_jobs = [j for j in scored if j.get("priority") != "HIGH"]

    if not high_jobs:
        _state["verified_jobs"] = scored
        return f"No HIGH jobs to verify. Passing {len(scored)} jobs through."

    log.info(f"Verifier: checking {len(high_jobs)} HIGH jobs via Groq...")

    VERIFY_PROMPT = (
        "Second-pass verification of HIGH jobs for Shashwat (CSE 2027 fresher, GenAI/ML/SDE).\n"
        "PERMISSIVE POLICY — keep HIGH unless there is an OBVIOUS disqualifier.\n"
        "Keep HIGH if: any AI/ML/LLM/backend/SDE work, product company, startup, research role.\n"
        "Keep HIGH if: internship at any recognisable tech company.\n"
        "Keep HIGH if: you are unsure — default is to keep.\n"
        "Downgrade to MEDIUM ONLY if: clearly non-technical (HR/Sales/Finance) OR pure IT outsourcing firm.\n"
        "SKIP ONLY if: scam-like post, unpaid internship, outside India + non-remote.\n"
        "Return ONLY JSON:\n"
        '[{"job_id":"...","verified_priority":"HIGH|MEDIUM|LOW|SKIP",'
        '"confidence":0-100,"reason":"one sentence"}]'
    )

    try:
        client = Groq(api_key=api_key)

        payload = [
            {
                "job_id":   j["job_id"],
                "title":    j["title"],
                "company":  j["company"],
                "job_type": j.get("job_type", "full-time"),   # ← pass type to verifier
                "reason":   (j.get("match_reason") or "")[:120],
            }
            for j in high_jobs
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": VERIFY_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Verify these {len(high_jobs)} HIGH jobs:\n"
                        f"{json.dumps(payload)}\n"
                        "JSON only."
                    ),
                },
            ],
            temperature=0.05,
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()

        results    = json.loads(raw)
        result_map = {r["job_id"]: r for r in results}
        downgraded = 0

        for job in high_jobs:
            v = result_map.get(job["job_id"])
            if not v:
                continue
            new_p = v.get("verified_priority", "HIGH")
            conf  = v.get("confidence", 100)

            # ── CHANGED: was conf > 50, now conf >= 85 ────────────────────
            # Only downgrade when the verifier is very sure it's wrong.
            # This was the root cause of 6/7 HIGH jobs getting dropped.
            if new_p != "HIGH" and conf >= 85:
                job["priority"]     = new_p
                job["match_reason"] = (
                    job.get("match_reason", "") +
                    f" [Verified: {v.get('reason','')}]"
                )
                downgraded += 1
                log.info(f"  Downgraded: {job['title']} @ {job['company']} → {job['priority']}")
            else:
                log.info(
                    f"  Confirmed HIGH: {job['title']} @ {job['company']} "
                    f"(conf={conf}%)"
                )

        log.info(f"Verifier done: {len(high_jobs)} checked, {downgraded} downgraded")

    except Exception as e:
        log.error(f"Verifier error: {e} — keeping all HIGH scores unchanged")

    all_verified = [j for j in high_jobs + other_jobs
                    if j.get("priority") not in ("SKIP", None)]
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_verified.sort(
        key=lambda j: (order.get(j.get("priority", "LOW"), 2),
                       -j.get("relevance_score", 0))
    )

    _state["verified_jobs"] = all_verified
    confirmed = len([j for j in all_verified if j.get("priority") == "HIGH"])
    return (
        f"Verified {len(high_jobs)} HIGH jobs. "
        f"Confirmed HIGH: {confirmed}. "
        f"Total for report: {len(all_verified)}."
    )


# ── Tool 4: Reporter ──────────────────────────────────────────────────────────

@tool("Build HTML report and send email digest")
def build_report_tool(dummy: str = "report") -> str:
    """
    Splits verified jobs into internships and full-time.
    Each group is ranked by score descending.
    Passes both lists to build_html_report() for separate sections.
    Email subject shows internship + job counts.
    """
    jobs = _state.get("verified_jobs", [])
    cfg  = _state.get("config", {})

    if not jobs:
        log.info("Reporter: no jobs to report.")
        return "No verified jobs — no report generated."

    # ── Split by job_type, sort by score within each group ─────────────────
    internships = sorted(
        [j for j in jobs if j.get("job_type") == "internship"],
        key=lambda j: -j.get("relevance_score", 0),
    )
    full_time = sorted(
        [j for j in jobs if j.get("job_type") == "full-time"],
        key=lambda j: -j.get("relevance_score", 0),
    )

    log.info(
        f"Reporter: {len(internships)} internships, {len(full_time)} full-time jobs"
    )

    today = date.today().isoformat()
    # ── build_html_report now receives two separate lists ──────────────────
    html  = build_html_report(internships, full_time, today)
    path  = REPORTS / f"report_{today}.html"
    path.write_text(html, encoding="utf-8")
    log.info(f"Report saved → {path}")

    if cfg.get("email_enabled"):
        subj = (
            f"Job Intel — {today} — "
            f"{len(internships)} internship{'s' if len(internships) != 1 else ''} · "
            f"{len(full_time)} job{'s' if len(full_time) != 1 else ''}"
        )
        send_email(html, cfg, subj)
        return (
            f"Report saved and emailed. "
            f"{len(internships)} internships, {len(full_time)} full-time jobs."
        )

    return f"Report saved to {path}. Email disabled in config."


# ── Crew builder ──────────────────────────────────────────────────────────────

def build_crew(cfg: dict) -> Crew:
    _state["config"] = cfg
    llm = _get_llm()

    scout = Agent(
        role="Job Scout",
        goal=(
            "Fetch all fresh job listings from all 20+ sources. "
            "Apply pre-LLM filters (URL fix, job-type classification, seniority drop, "
            "experience filter) BEFORE handing off to the Analyst."
        ),
        backstory="Expert at sourcing and cleaning tech job data across all platforms.",
        tools=[fetch_all_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Job Analyst",
        goal="Score pre-filtered fresh jobs by intent match using Groq LLM.",
        backstory="Technical recruiter scoring GenAI/ML/SDE roles for a 2027 fresher.",
        tools=[score_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    verifier = Agent(
        role="Job Verifier",
        goal=(
            "Permissive second-pass on HIGH priority jobs. "
            "Keep HIGH when uncertain. Only downgrade with >= 85% confidence."
        ),
        backstory="Senior recruiter doing a light quality check — defaults to keeping HIGH.",
        tools=[verify_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Job Reporter",
        goal=(
            "Split jobs into Top Internships and Top Full-Time Jobs. "
            "Build HTML digest with two ranked sections. Send email."
        ),
        backstory="Turns verified job scores into a clean daily briefing with direct apply links.",
        tools=[build_report_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task_scout = Task(
        description=(
            "Call fetch_all_jobs_tool. Fetch all sources, deduplicate, "
            "run pre-LLM filters. Report counts at each stage."
        ),
        expected_output="Total fetched, fresh after dedup, after filter. Internship vs full-time split.",
        agent=scout,
    )

    task_analyse = Task(
        description="Call score_jobs_tool to score all filtered fresh jobs with Groq.",
        expected_output="Count of HIGH, MEDIUM, LOW jobs and how many were dropped.",
        agent=analyst,
        context=[task_scout],
    )

    task_verify = Task(
        description=(
            "Call verify_jobs_tool. Be permissive — only downgrade HIGH jobs "
            "when confidence is >= 85%."
        ),
        expected_output="Count of confirmed HIGH jobs and any downgrades made.",
        agent=verifier,
        context=[task_analyse],
    )

    task_report = Task(
        description=(
            "Call build_report_tool. Split into internships and full-time. "
            "Save report and send email with internship + job counts in subject."
        ),
        expected_output="Internship count, full-time count, report path, email status.",
        agent=reporter,
        context=[task_verify],
    )

    return Crew(
        agents=[scout, analyst, verifier, reporter],
        tasks=[task_scout, task_analyse, task_verify, task_report],
        process=Process.sequential,
        verbose=True,
    )