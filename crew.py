"""
crew.py
CrewAI with Ollama backend — no rate limits, no cache_breakpoint issues.
4-stage pipeline: Scout → Analyst → Verifier → Reporter
"""
from groq import Groq
import json
import logging
from pathlib import Path
from datetime import date
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from sources.public_apis    import fetch_remotive, fetch_arbeitnow, fetch_jobicy
from sources.career_portals import fetch_all_career_portals
from sources.linkedin       import fetch_linkedin
from sources.naukri         import fetch_naukri
from scorer                 import score_jobs_with_llm
from reporter               import build_html_report, send_email
from utils                  import load_seen, save_seen, deduplicate

log = logging.getLogger(__name__)

BASE    = Path(__file__).parent
SEEN    = BASE / "logs" / "seen_jobs.json"
REPORTS = BASE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
(BASE / "logs").mkdir(parents=True, exist_ok=True)

_state: dict = {
    "raw_jobs":      [],
    "fresh_jobs":    [],
    "scored_jobs":   [],
    "verified_jobs": [],
    "config":        {},
}


def _get_llm() -> LLM:
    """Ollama LLM — local, free, no rate limits, no cache issues."""
    return LLM(
        model="ollama/qwen3:4b",
        base_url="http://localhost:11434",
        temperature=0.1,
    )


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("Fetch all jobs from all sources")
def fetch_all_jobs_tool(dummy: str = "go") -> str:
    """
    Fetches all job listings from 15 sources: public APIs, LinkedIn,
    Naukri, and company career portals. Deduplicates against seen jobs.
    Returns count summary.
    """
    cfg      = _state["config"]
    keywords = cfg.get("search_keywords", [])
    location = cfg.get("location", "India")

    raw = []
    raw.extend(fetch_remotive(keywords))
    raw.extend(fetch_arbeitnow(keywords))
    raw.extend(fetch_jobicy(keywords))
    raw.extend(fetch_linkedin(keywords, location))
    raw.extend(fetch_naukri())
    raw.extend(fetch_all_career_portals())

    seen = load_seen(SEEN)
    fresh, seen = deduplicate(raw, seen)
    save_seen(SEEN, seen)

    _state["raw_jobs"]   = raw
    _state["fresh_jobs"] = fresh

    log.info(f"Scout: {len(raw)} total, {len(fresh)} fresh")
    return f"Fetched {len(raw)} jobs. Fresh after dedup: {len(fresh)}."


@tool("Score jobs with LLM intent matching")
def score_jobs_tool(dummy: str = "score") -> str:
    """
    Scores fresh jobs using Groq llama-3.3-70b-versatile.
    Drops irrelevant jobs. Returns scoring summary.
    """
    fresh      = _state.get("fresh_jobs", [])
    cfg        = _state.get("config", {})
    api_key    = cfg.get("groq_api_key", "")
    batch_size = cfg.get("llm_batch_size", 15)

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
        f"Dropped: {len(fresh) - len(scored)}."
    )


@tool("Verify HIGH priority jobs for accuracy")
def verify_jobs_tool(dummy: str = "verify") -> str:
    """
    Re-scores all HIGH priority jobs using Groq for second-pass accuracy check.
    Downgrades any that don't hold up.
    """
    scored = _state.get("scored_jobs", [])
    cfg    = _state.get("config", {})
    api_key = cfg.get("groq_api_key", "")

    if not scored:
        _state["verified_jobs"] = []
        return "No scored jobs to verify."

    high_jobs  = [j for j in scored if j.get("priority") == "HIGH"]
    other_jobs = [j for j in scored if j.get("priority") != "HIGH"]

    if not high_jobs:
        _state["verified_jobs"] = scored
        return f"No HIGH jobs to verify. Passing {len(scored)} jobs through."

    log.info(f"Verifier: re-checking {len(high_jobs)} HIGH jobs via Groq...")

    VERIFY_PROMPT = """
You are a strict senior recruiter doing a SECOND OPINION on jobs already scored HIGH.
Be MORE critical than the first pass.

Candidate: Sam, B.Tech CS 2027 fresher, GenAI intern, YCCE Nagpur.
Skills: LangChain, RAG, ChromaDB, XGBoost, FastAPI, Python, Java, AWS S3, Docker.
Wants: GenAI/ML/SDE at product companies or AI startups. Fresher only (0-2 yrs).
Vetoes: TCS/Infosys/Wipro/Accenture/HCL (unless explicit AI/ML title),
        3+ years required, pure DevOps/hardware/sales, outside India.

For each job answer:
1. Does this GENUINELY involve AI/ML/backend engineering? (not just mentions it)
2. Is this a product company or AI startup? (not IT services)
3. Is this truly open to freshers graduating 2027?

ALL THREE yes → keep HIGH.
Any doubt → downgrade to MEDIUM.
Clearly wrong → SKIP.

Return ONLY valid JSON array:
[{"job_id":"...","verified_priority":"HIGH|MEDIUM|LOW|SKIP","confidence":0-100,"reason":"one sentence"}]
"""

    try:
        client = Groq(api_key=api_key)

        payload = [
            {
                "job_id":       j["job_id"],
                "title":        j["title"],
                "company":      j["company"],
                "description":  (j.get("description") or "")[:400],
                "match_reason": j.get("match_reason", ""),
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
                        f"{json.dumps(payload, indent=2)}\n"
                        "Return JSON array only."
                    ),
                },
            ],
            temperature=0.05,
            max_tokens=1024,
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
            if v:
                new_p = v.get("verified_priority", "HIGH")
                conf  = v.get("confidence", 100)
                if new_p != "HIGH" or conf < 75:
                    job["priority"]     = new_p
                    job["match_reason"] += f" [Verified: {v.get('reason','')}]"
                    downgraded += 1
                    log.info(f"  Downgraded: {job['title']} @ {job['company']} → {new_p}")
                else:
                    log.info(f"  Confirmed HIGH: {job['title']} @ {job['company']} ({conf}%)")

        log.info(f"Verifier done: {downgraded} downgraded")

    except Exception as e:
        log.error(f"Verifier error: {e} — keeping original scores")

    all_verified = [j for j in high_jobs + other_jobs if j.get("priority") != "SKIP"]
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_verified.sort(key=lambda j: (order.get(j.get("priority", "LOW"), 2),
                                      -j.get("relevance_score", 0)))

    _state["verified_jobs"] = all_verified
    confirmed = len([j for j in all_verified if j.get("priority") == "HIGH"])
    return (
        f"Verified {len(high_jobs)} HIGH jobs. "
        f"Confirmed: {confirmed}. Total passing: {len(all_verified)}."
    )


@tool("Build report and send email")
def build_report_tool(dummy: str = "report") -> str:
    """
    Builds styled HTML report from verified jobs and sends email digest.
    Returns status message.
    """
    jobs = _state.get("verified_jobs", [])
    cfg  = _state.get("config", {})

    if not jobs:
        return "No verified jobs — no report generated."

    today = date.today().isoformat()
    html  = build_html_report(jobs, today)
    path  = REPORTS / f"report_{today}.html"
    path.write_text(html, encoding="utf-8")
    log.info(f"Report saved → {path}")

    if cfg.get("email_enabled"):
        high = len([j for j in jobs if j.get("priority") == "HIGH"])
        send_email(html, cfg, f"Job Intel: {len(jobs)} matches ({high} HIGH) — {today}")
        return f"Report saved and emailed. {len(jobs)} jobs, {high} HIGH."

    return f"Report saved to {path}. Email disabled."


# ── Crew builder ──────────────────────────────────────────────────────────────

def build_crew(cfg: dict) -> Crew:
    _state["config"] = cfg
    llm = _get_llm()

    scout = Agent(
        role="Job Scout",
        goal="Fetch all fresh job listings from all sources.",
        backstory="Expert at sourcing tech jobs across all Indian job platforms.",
        tools=[fetch_all_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Job Analyst",
        goal="Score fresh jobs by intent match. Drop irrelevant ones ruthlessly.",
        backstory="Technical recruiter scoring GenAI/ML/SDE roles for a 2027 fresher.",
        tools=[score_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    verifier = Agent(
        role="Job Verifier",
        goal="Double-check all HIGH priority jobs. Downgrade any that are inaccurate.",
        backstory="Senior recruiter doing a strict second-pass quality check on top jobs.",
        tools=[verify_jobs_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Job Reporter",
        goal="Build the HTML digest and send it.",
        backstory="Communicator turning verified job scores into a clean daily email.",
        tools=[build_report_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task_scout = Task(
        description="Call fetch_all_jobs_tool to fetch and deduplicate all jobs.",
        expected_output="Count of total and fresh jobs fetched.",
        agent=scout,
    )

    task_analyse = Task(
        description="Call score_jobs_tool to score all fresh jobs.",
        expected_output="Count of HIGH, MEDIUM, LOW jobs and how many dropped.",
        agent=analyst,
        context=[task_scout],
    )

    task_verify = Task(
        description="Call verify_jobs_tool to double-check all HIGH priority jobs.",
        expected_output="Count of confirmed HIGH jobs and any downgrades.",
        agent=verifier,
        context=[task_analyse],
    )

    task_report = Task(
        description="Call build_report_tool to save and email the final report.",
        expected_output="Confirmation that report was saved and email sent.",
        agent=reporter,
        context=[task_verify],
    )

    return Crew(
        agents=[scout, analyst, verifier, reporter],
        tasks=[task_scout, task_analyse, task_verify, task_report],
        process=Process.sequential,
        verbose=True,
    )