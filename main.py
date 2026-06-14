"""
main.py — CrewAI pipeline with Ollama backend.
"""

import os
import logging
from datetime import date
from pathlib import Path

from utils import setup_logging, load_config, load_profile
from crew  import build_crew

BASE   = Path(__file__).parent
LOGS   = BASE / "logs" / "agent.log"
CONFIG = BASE / "config" / "config.json"

(BASE / "logs").mkdir(parents=True, exist_ok=True)
(BASE / "reports").mkdir(parents=True, exist_ok=True)

setup_logging(LOGS)
log = logging.getLogger(__name__)

DEFAULT_KEYWORDS = [
    "generative AI engineer fresher",
    "generative AI intern",
    "LLM engineer entry level",
    "LLM engineer intern",
    "AI engineer 2026 2027 batch",
    "AI engineer intern",
    "machine learning engineer fresher",
    "machine learning intern",
    "software engineer AI india",
    "software engineer intern india",
    "backend engineer AI startup india",
    "backend developer intern python",
    "SDE fresher 2027 india",
    "SDE intern india",
    "data science intern india",
]


def _default_config() -> dict:
    return {
        "groq_api_key":    os.getenv("GROQ_API_KEY", ""),
        "email_enabled":   os.getenv("EMAIL_ENABLED", "false").lower() == "true",
        "email_from":      os.getenv("EMAIL_FROM", ""),
        "email_to":        os.getenv("EMAIL_TO", ""),
        "smtp_host":       os.getenv("SMTP_HOST", "smtp-relay.brevo.com"),
        "smtp_port":       int(os.getenv("SMTP_PORT", "587")),
        "smtp_user":       os.getenv("SMTP_USER", ""),
        "smtp_pass":       os.getenv("SMTP_PASS", ""),
        "location":        os.getenv("LOCATION", "India"),
        "llm_batch_size":  10,
        "search_keywords": DEFAULT_KEYWORDS,
        "include_internships": True,
    }


def main():
    log.info("=" * 50)
    log.info(f"Job Intel Agent — {date.today().isoformat()}")
    log.info("=" * 50)

    try:
        cfg = load_config(CONFIG)
    except FileNotFoundError:
        log.warning("config/config.json not found — using env vars and defaults")
        cfg = _default_config()

    profile = load_profile()
    if profile:
        cfg["profile"] = profile
        log.info(f"Profile loaded: {profile.get('name', 'unknown')} ({profile.get('graduation_batch', '')} batch)")
    else:
        log.warning("config/profile.json not found — run: python setup_profile.py")

    if not cfg.get("search_keywords"):
        cfg["search_keywords"] = DEFAULT_KEYWORDS

    if cfg.get("include_internships", True):
        intern_kw = [k for k in cfg["search_keywords"] if "intern" in k.lower()]
        if len(intern_kw) < 3:
            cfg["search_keywords"] = list(dict.fromkeys(
                cfg["search_keywords"] + [k for k in DEFAULT_KEYWORDS if "intern" in k.lower()]
            ))

    log.info("Building crew...")
    crew = build_crew(cfg)

    log.info("Running crew...")
    result = crew.kickoff()

    log.info(f"Result: {result}")
    log.info("Run complete.")


if __name__ == "__main__":
    main()
