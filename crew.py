"""
crew.py — CrewAI 1.x pipeline.
Agents use Ollama qwen3:4b (local, no rate limits, tiny token usage).
Scoring and verification use Groq llama-3.3-70b (accurate, fast, free tier).
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

log = logging.getLogger(__name__)

BASE    = Path(__file__).parent
SEEN    = BASE / "logs" / "seen_jobs.json"
REPORTS = BASE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
(BASE / "logs").mkdir(parents=True, exist_ok=True)

# Shared state across tools in one run
_state: dict = {
    "raw_jobs":      [],
    "fresh_jobs":    [],
    "scored_jobs":   [],
    "verified_jobs": [],
    "config":        {},
}


def _get_llm() -> LLM:
    """
    Ollama for agent reasoning — local, free, no rate limits.
    Only used for the 'think and call tool' step, not for scoring.
    Token usage: ~200-300 tokens per agent call.
    """
    return LLM(
        model="ollama/qwen3:4b",
        base_url="http://localhost:11434",
        temperature=0.1,
    )


# ── Tool 1: Scout ─────────────────────────────────────────────────────────────

@tool("Fetch all jobs from all sources")
def fetch_all_jobs_tool(dummy: str = "go") -> str:
    """
    Fetches all job listings from all sources: public APIs (Remotive, Arbeitnow,
    Jobicy, Himalayas, Freshersworld), LinkedIn, Naukri, and 16 company career
    portals. Deduplicates against seen_jobs.json (7-day expiry).
    Returns count summary.
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

    _state["raw_jobs"]   = raw
    _state["fresh_jobs"] = fresh

    log.info(f"Scout: {len(raw)} total, {len(fresh)} fresh after dedup")
    return f"Fetched {len(raw)} total jobs. Fresh after dedup: {len(fresh)}."


# ── Tool 2: Analyst ───────────────────────────────────────────────────────────

@tool("Score jobs with Groq LLM intent matching")
def score_jobs_tool(dummy: str = "score") -> str:
    """
    Scores fresh jobs using Groq llama-3.3-70b with intent-based matching.
    Uses compressed candidate profile (~60 tokens) to minimize API token usage.
    Drops irrelevant jobs (SKIP). Returns scoring summary.
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
    Second-pass accuracy check on HIGH priority jobs using Groq.
    Uses full candidate profile for detailed verification.
    Downgrades any HIGH jobs that don't hold up on re-examination.
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

    # Compressed verify prompt — uses full profile for accuracy
    VERIFY_PROMPT = (
        "Second-pass verification of HIGH jobs for Sam (CSE 2027 fresher, GenAI/ML/SDE target).\n"
        "Keep HIGH if the role involves any of: AI, ML, LLM, backend engineering, data science, "
        "software engineering at a product company or startup.\n"
        "Downgrade to MEDIUM only if role is clearly NOT technical or is at an IT outsourcing firm.\n"
        "NEVER downgrade for vague reasons — if in doubt, keep HIGH.\n"
        "Internships with AI/ML content = HIGH. Research roles = HIGH. SDE at good company = HIGH.\n"
        "Only SKIP if: pure sales, pure HR, pure DevOps with zero coding, or obvious outsourcing firm.\n"
        "Return ONLY JSON: "
        '[{"job_id":"...","verified_priority":"HIGH|MEDIUM|LOW|SKIP",'
        '"confidence":0-100,"reason":"one sentence"}]'
    )

    try:
        client = Groq(api_key=api_key)

        # Minimal payload for verification
        payload = [
            {
                "job_id":  j["job_id"],
                "title":   j["title"],
                "company": j["company"],
                "reason":  (j.get("match_reason") or "")[:120],
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
            if new_p != "HIGH" and conf > 50:
                job["priority"]     = new_p if new_p != "HIGH" else "MEDIUM"
                job["match_reason"] = (
                    job.get("match_reason", "") +
                    f" [Verified: {v.get('reason','')}]"
                )
                downgraded += 1
                log.info(f"  Downgraded: {job['title']} @ {job['company']} → {job['priority']}")
            else:
                log.info(
                    f"  Confirmed HIGH: {job['title']} @ {job['company']} "
                    f"({conf}% confidence)"
                )

        log.info(f"Verifier done: {len(high_jobs)} checked, {downgraded} downgraded")

    except Exception as e:
        log.error(f"Verifier error: {e} — keeping original HIGH scores")

    # Merge and sort
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
        f"Total passing report: {len(all_verified)}."
    )


# ── Tool 4: Reporter ──────────────────────────────────────────────────────────

@tool("Build HTML report and send email digest")
def build_report_tool(dummy: str = "report") -> str:
    """
    Builds a styled HTML report from verified jobs, saves it to reports/ folder,
    and sends it via Brevo SMTP if email_enabled is true in config.
    Returns status message.
    """
    jobs = _state.get("verified_jobs", [])
    cfg  = _state.get("config", {})

    if not jobs:
        log.info("Reporter: no jobs to report.")
        return "No verified jobs — no report generated."

    today = date.today().isoformat()
    html  = build_html_report(jobs, today)
    path  = REPORTS / f"report_{today}.html"
    path.write_text(html, encoding="utf-8")
    log.info(f"Report saved → {path}")

    if cfg.get("email_enabled"):
        high = len([j for j in jobs if j.get("priority") == "HIGH"])
        subj = f"Job Intel: {len(jobs)} matches ({high} HIGH) — {today}"
        send_email(html, cfg, subj)
        return f"Report saved and emailed. {len(jobs)} jobs, {high} HIGH."

    return f"Report saved to {path}. Email disabled in config."


# ── Crew builder ──────────────────────────────────────────────────────────────

def build_crew(cfg: dict) -> Crew:
    _state["config"] = cfg
    llm = _get_llm()

    scout = Agent(
        role="Job Scout",
        goal="Fetch all fresh job listings from all 20+ sources.",
        backstory="Expert at sourcing tech jobs across all Indian and global platforms.",
        tools=[fetch_all_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Job Analyst",
        goal="Score fresh jobs by intent match using Groq LLM. Drop irrelevant ones.",
        backstory="Technical recruiter scoring GenAI/ML/SDE roles for a 2027 fresher.",
        tools=[score_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    verifier = Agent(
        role="Job Verifier",
        goal="Double-check all HIGH priority jobs. Downgrade any that are inaccurate.",
        backstory="Senior recruiter doing strict second-pass quality check on top jobs.",
        tools=[verify_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Job Reporter",
        goal="Build the HTML digest and send the daily email.",
        backstory="Communicator turning verified job scores into a clean daily briefing.",
        tools=[build_report_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task_scout = Task(
        description="Call fetch_all_jobs_tool to fetch and deduplicate all jobs.",
        expected_output="Count of total jobs fetched and fresh jobs after dedup.",
        agent=scout,
    )

    task_analyse = Task(
        description="Call score_jobs_tool to score all fresh jobs with Groq.",
        expected_output="Count of HIGH, MEDIUM, LOW jobs and how many dropped.",
        agent=analyst,
        context=[task_scout],
    )

    task_verify = Task(
        description="Call verify_jobs_tool to double-check all HIGH priority jobs.",
        expected_output="Count of confirmed HIGH jobs and any downgrades made.",
        agent=verifier,
        context=[task_analyse],
    )

    task_report = Task(
        description="Call build_report_tool to save report and send email digest.",
        expected_output="Status confirming report saved and email sent or disabled.",
        agent=reporter,
        context=[task_verify],
    )

    return Crew(
        agents=[scout, analyst, verifier, reporter],
        tasks=[task_scout, task_analyse, task_verify, task_report],
        process=Process.sequential,
        verbose=True,
    )