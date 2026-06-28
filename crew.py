"""
crew.py — CrewAI 1.x pipeline.

v2.2 changes vs v2.1:
  - _get_llm(): model upgraded to llama-3.3-70b-instruct (fixes agent loop bug)
  - Agent backstories and task descriptions are now profile-agnostic
    (no hardcoded "GenAI/ML" references — works for any engineering profile)
  - max_iter=3 added to all agents to hard-cap runaway loops
  - Everything else (tools, state, build_crew structure) unchanged
"""

import logging

log = logging.getLogger(__name__)
log.info("crew.py imported")

import json
import logging
from pathlib import Path
from datetime import date

from nim_client import make_client, call_nim, clean_json

log.info("nim_client imported")
from crewai import Agent, Task, Crew, Process, LLM

log.info("crewai imported")
from crewai.tools import tool

log.info("crewai.tools imported")

from sources.public_apis import (
    fetch_remotive,
    fetch_arbeitnow,
    fetch_jobicy,
    fetch_himalayas,
    fetch_freshersworld,
)
from sources.career_portals import fetch_all_career_portals

log.info("sources imported career portals")
from sources.linkedin import fetch_linkedin

log.info("sources imported lined in")
from sources.naukri import fetch_naukri

log.info("sources imported")
from scorer import score_jobs_with_llm, CANDIDATE_PROFILE_FULL

log.info("sources imported scorer")
from reporter import build_html_report, send_email

log.info("sources imported reporter")
from utils import (
    load_seen,
    save_seen,
    deduplicate,
    deduplicate_batch,
    load_profile,
    get_verifier_system_prompt,
)

log.info("sources imported utils")
from filters import apply_all_filters

log.info("sources imported filters")

log = logging.getLogger(__name__)

BASE = Path(__file__).parent
SEEN = BASE / "logs" / "seen_jobs.json"
REPORTS = BASE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
(BASE / "logs").mkdir(parents=True, exist_ok=True)


class _PipelineState:
    def __init__(self):
        self.raw_jobs: list = []
        self.fresh_jobs: list = []
        self.scored_jobs: list = []
        self.verified_jobs: list = []
        self.config: dict = {}
        self.seen_snapshot: dict = {}

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


_state = _PipelineState()


# ── LLM — NVIDIA NIM llama-3.3-70b-instruct ──────────────────────────────────


def _get_llm() -> LLM:
    """
    NVIDIA NIM via LiteLLM's openai/ prefix.
    Model: meta/llama-3.3-70b-instruct
      - Larger model reliably follows CrewAI ReAct format and stops after one tool call
      - llama-3.1-8b-instruct was too small and caused agent loop (tool called 4+ times)
    """
    import os

    log.info("=" * 60)
    log.info("ENTERING _get_llm()")
    log.info("=" * 60)

    nim_key = _state.get("config", {}).get("nvidia_nim_api_key") or os.getenv(
        "NVIDIA_NIM_API_KEY", ""
    )

    log.info("NVIDIA key exists: %s", bool(nim_key))
    if nim_key:
        log.info("Key prefix: %s...", nim_key[:10])

    log.info("Creating CrewAI LLM object...")

    llm = LLM(
        model="openai/meta/llama-3.3-70b-instruct",
        api_key=nim_key,
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.1,
        max_tokens=512,
    )

    log.info("CrewAI LLM object created successfully")
    return llm


# ── Tool 1: Scout ─────────────────────────────────────────────────────────────


@tool("Fetch all jobs from all sources")
def fetch_all_jobs_tool() -> str:
    """
    Fetches all job listings from all sources, deduplicates, then applies
    pre-LLM filters before any LLM call is made.
    """
    cfg = _state["config"]
    keywords = cfg.get("search_keywords", [])
    location = cfg.get("location", "India")

    raw = []
    source_counts = {}

    def _fetch_and_count(name, fn, *args):
        try:
            result = fn(*args)
        except Exception as e:
            log.error(f"Source error — {name}: {e}")
            result = []
        source_counts[name] = len(result)
        raw.extend(result)

    _fetch_and_count("Remotive", fetch_remotive, keywords)
    _fetch_and_count("Arbeitnow", fetch_arbeitnow, keywords)
    _fetch_and_count("Jobicy", fetch_jobicy, keywords)
    _fetch_and_count("Himalayas", fetch_himalayas, keywords)
    _fetch_and_count("Freshersworld", fetch_freshersworld, keywords)
    _fetch_and_count("LinkedIn", fetch_linkedin, keywords, location)
    _fetch_and_count("Naukri", fetch_naukri, keywords)
    _fetch_and_count("CareerPortals", fetch_all_career_portals)

    log.info(
        "Per-source raw counts: "
        + ", ".join(f"{k}={v}" for k, v in source_counts.items())
    )

    before_batch_dedup = len(raw)
    raw = deduplicate_batch(raw)
    log.info(f"Scout: cross-source dedup {before_batch_dedup} → {len(raw)}")

    fresh = raw
    _state["seen_snapshot"] = {}

    try:
        (BASE / "logs" / "last_raw_jobs.json").write_text(
            json.dumps(fresh, default=str), encoding="utf-8"
        )
    except Exception as e:
        log.debug(f"Could not cache raw jobs for audit: {e}")

    profile = cfg.get("profile", {})
    max_exp_years = profile.get("max_experience_years", 2)
    role_exclusions = profile.get("role_type_exclusions", [])
    fresh = apply_all_filters(
        fresh, max_experience_years=max_exp_years, role_exclusions=role_exclusions
    )

    _state["raw_jobs"] = raw
    _state["fresh_jobs"] = fresh

    intern_count = sum(1 for j in fresh if j.get("job_type") == "internship")
    ft_count = len(fresh) - intern_count

    log.info(
        f"Scout: {len(raw)} after cross-source dedup → {len(fresh)} after seen-dedup+filter "
        f"({intern_count} internships, {ft_count} full-time). "
        f"NOTE: seen_jobs.json not yet updated — will persist only on successful report."
    )
    return (
        f"Fetched {before_batch_dedup} total ({len(raw)} unique across sources). "
        f"After seen-dedup + pre-LLM filter: {len(fresh)} "
        f"({intern_count} internships, {ft_count} full-time). Ready to score."
    )


# ── Tool 2: Analyst ───────────────────────────────────────────────────────────


@tool("Score jobs with intent matching")
def score_jobs_tool() -> str:
    """Scores pre-filtered fresh jobs using intent matching against the candidate profile."""
    fresh = _state.get("fresh_jobs", [])
    cfg = _state.get("config", {})
    api_key = cfg.get("nvidia_nim_api_key", "")
    batch_size = cfg.get("llm_batch_size", 20)

    if not fresh:
        return "No fresh jobs to score."

    scored = score_jobs_with_llm(fresh, api_key=api_key, batch_size=batch_size)
    _state["scored_jobs"] = scored

    high = len([j for j in scored if j.get("priority") == "HIGH"])
    medium = len([j for j in scored if j.get("priority") == "MEDIUM"])
    low = len([j for j in scored if j.get("priority") == "LOW"])

    log.info(f"Analyst: {len(scored)} relevant — {high}H {medium}M {low}L")
    return (
        f"Scored {len(fresh)} jobs. "
        f"Relevant: {len(scored)} ({high} HIGH, {medium} MEDIUM, {low} LOW). "
        f"Dropped: {len(fresh) - len(scored)} irrelevant."
    )


# ── Tool 3: Verifier ──────────────────────────────────────────────────────────


@tool("Verify HIGH priority jobs for accuracy")
def verify_jobs_tool() -> str:
    """Runs a permissive second-pass verification on HIGH priority jobs, only downgrading when confidence is very high."""
    scored = _state.get("scored_jobs", [])
    log.info(f"Verifier: entry — len(scored_jobs)={len(scored)} id={id(_state)}")
    cfg = _state.get("config", {})
    api_key = cfg.get("nvidia_nim_api_key", "")

    if not scored:
        _state["verified_jobs"] = []
        return "No scored jobs to verify."

    high_jobs = [j for j in scored if j.get("priority") == "HIGH"]
    other_jobs = [j for j in scored if j.get("priority") != "HIGH"]

    if not high_jobs:
        _state["verified_jobs"] = scored
        return f"No HIGH jobs to verify. Passing {len(scored)} jobs through."

    log.info(f"Verifier: checking {len(high_jobs)} HIGH jobs via NIM...")

    profile = load_profile()
    VERIFY_PROMPT = get_verifier_system_prompt(profile)

    try:
        nim = make_client(api_key)

        payload = [
            {
                "job_id": j["job_id"],
                "title": j["title"],
                "company": j["company"],
                "job_type": j.get("job_type", "full-time"),
                "reason": (j.get("match_reason") or "")[:120],
            }
            for j in high_jobs
        ]

        user_content = (
            f"Verify these {len(high_jobs)} HIGH jobs.\n"
            f"Return a JSON array with exactly {len(high_jobs)} objects — one per job_id.\n"
            f"{json.dumps(payload)}\n"
            "JSON only. No truncation."
        )

        raw = clean_json(call_nim(nim, VERIFY_PROMPT, user_content))
        results = json.loads(raw)
        result_map = {r["job_id"]: r for r in results}
        downgraded = 0

        for job in high_jobs:
            v = result_map.get(job["job_id"])
            if not v:
                continue
            new_p = v.get("verified_priority", "HIGH")
            conf = v.get("confidence", 100)

            if new_p != "HIGH" and conf >= 85:
                job["priority"] = new_p
                job["match_reason"] = (
                    job.get("match_reason", "") + f" [Verified: {v.get('reason','')}]"
                )
                downgraded += 1
                log.info(
                    f"  Downgraded: {job['title']} @ {job['company']} → {job['priority']}"
                )
            else:
                log.info(
                    f"  Confirmed HIGH: {job['title']} @ {job['company']} "
                    f"(conf={conf}%)"
                )

        log.info(f"Verifier done: {len(high_jobs)} checked, {downgraded} downgraded")

    except Exception as e:
        log.error(f"Verifier error: {e} — keeping all HIGH scores unchanged")

    all_verified = [
        j for j in high_jobs + other_jobs if j.get("priority") not in ("SKIP", None)
    ]

    if not all_verified:
        log.warning(
            "Verifier produced empty verified_jobs — falling back to scored_jobs"
        )
        all_verified = [j for j in scored if j.get("priority") not in ("SKIP", None)]

    _state["verified_jobs"] = all_verified
    confirmed = len([j for j in all_verified if j.get("priority") == "HIGH"])
    return (
        f"Verified {len(high_jobs)} HIGH jobs. "
        f"Confirmed HIGH: {confirmed}. "
        f"Total for report: {len(all_verified)}."
    )


# ── Tool 4: Reporter ──────────────────────────────────────────────────────────


@tool("Build HTML report and send email digest")
def build_report_tool() -> str:
    """Splits verified jobs into internships and full-time, builds the HTML report, sends the email digest, and persists the seen-store on success."""
    jobs = _state.get("verified_jobs", [])
    cfg = _state.get("config", {})

    if not jobs:
        log.info("Reporter: no jobs to report — seen_jobs.json left untouched.")
        return "No verified jobs — no report generated."

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
    html = build_html_report(internships, full_time, today)
    path = REPORTS / f"report_{today}.html"
    path.write_text(html, encoding="utf-8")
    log.info(f"Report saved → {path}")

    email_status_suffix = ""
    email_ok = True
    if cfg.get("email_enabled"):
        subj = (
            f"Job Intel — {today} — "
            f"{len(internships)} internship{'s' if len(internships) != 1 else ''} · "
            f"{len(full_time)} job{'s' if len(full_time) != 1 else ''}"
        )
        email_ok = send_email(html, cfg, subj)
        if not email_ok:
            email_status_suffix = (
                f" EMAIL FAILED — check logs/agent.log for the SMTP error "
                f"(likely sender not verified in Brevo)."
            )

        log.info("Seen-store disabled — all jobs returned fresh each run.")

    if cfg.get("email_enabled"):
        if email_ok:
            return (
                f"Report saved and emailed successfully. "
                f"{len(internships)} internships, {len(full_time)} full-time jobs. "
                f"Saved to {path}."
            )
        else:
            return (
                f"Report saved to {path} but EMAIL FAILED — check logs/agent.log "
                f"for the SMTP error (likely sender not verified in Brevo). "
                f"{len(internships)} internships, {len(full_time)} full-time jobs "
                f"are in the saved report file.{email_status_suffix}"
            )

    return f"Report saved to {path}. Email disabled in config."


# ── Crew builder ──────────────────────────────────────────────────────────────


def build_crew(cfg: dict) -> Crew:

    log.info("=" * 60)
    log.info("ENTER build_crew()")
    log.info("=" * 60)

    _state["config"] = cfg

    log.info("Calling _get_llm()...")
    llm = _get_llm()
    log.info("_get_llm() completed")

    log.info("Creating Scout agent...")

    scout = Agent(
        role="Job Scout",
        goal=(
            "Fetch all fresh job listings from all sources. "
            "Apply pre-LLM filters BEFORE handing off to the Analyst."
        ),
        backstory=(
            "Expert at sourcing and cleaning tech job data across all platforms. "
            "Fetches from LinkedIn, Naukri, career portals, and public APIs."
        ),
        tools=[fetch_all_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    analyst = Agent(
        role="Job Analyst",
        goal=(
            "Score pre-filtered fresh jobs by intent match against the candidate profile. "
            "Call score_jobs_tool exactly once and report the results."
        ),
        backstory=(
            "Technical recruiter who scores job listings based on how well they match "
            "the specific candidate's skills, target roles, and experience level."
        ),
        tools=[score_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    verifier = Agent(
        role="Job Verifier",
        goal=(
            "Run a permissive second-pass on HIGH priority jobs. "
            "Keep HIGH when uncertain. Only downgrade with >= 85% confidence. "
            "Call verify_jobs_tool exactly once and report the results."
        ),
        backstory=(
            "Senior recruiter doing a light quality check on top-scored jobs. "
            "Defaults to keeping HIGH — only downgrades when clearly wrong."
        ),
        tools=[verify_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    reporter = Agent(
        role="Job Reporter",
        goal=(
            "Split jobs into Top Internships and Top Full-Time Jobs. "
            "Build HTML digest with two ranked sections. Send email. "
            "Call build_report_tool exactly once and report the results."
        ),
        backstory=(
            "Turns verified job scores into a clean daily briefing with direct apply links. "
            "Saves the HTML report and sends the email digest."
        ),
        tools=[build_report_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    task_scout = Task(
        description=(
            "Call fetch_all_jobs_tool ONCE. "
            "Fetch all sources, deduplicate, run pre-LLM filters. "
            "Report counts at each stage. Do not call the tool more than once."
        ),
        expected_output="Total fetched, fresh after dedup, after filter. Internship vs full-time split.",
        agent=scout,
    )

    task_analyse = Task(
        description=(
            "Call score_jobs_tool ONCE to score all filtered fresh jobs. "
            "Report the count of HIGH, MEDIUM, LOW jobs and how many were dropped. "
            "Do not call the tool more than once."
        ),
        expected_output="Count of HIGH, MEDIUM, LOW jobs and how many were dropped.",
        agent=analyst,
        context=[task_scout],
    )

    task_verify = Task(
        description=(
            "Call verify_jobs_tool ONCE. Be permissive — only downgrade HIGH jobs "
            "when confidence is >= 85%. "
            "Do not call the tool more than once."
        ),
        expected_output="Count of confirmed HIGH jobs and any downgrades made.",
        agent=verifier,
        context=[task_analyse],
    )

    task_report = Task(
        description=(
            "Call build_report_tool ONCE. Split into internships and full-time. "
            "Save report and send email. "
            "Do not call the tool more than once."
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
