"""
main.py — CrewAI pipeline with Ollama backend.
"""

import os
import json
import logging
from datetime import date
from pathlib import Path

from utils import setup_logging, load_config
from crew  import build_crew

BASE   = Path(__file__).parent
LOGS   = BASE / "logs" / "agent.log"
CONFIG = BASE / "config" / "config.json"

(BASE / "logs").mkdir(parents=True, exist_ok=True)
(BASE / "reports").mkdir(parents=True, exist_ok=True)

setup_logging(LOGS)
log = logging.getLogger(__name__)


def main():
    log.info("=" * 50)
    log.info(f"Job Intel Agent — {date.today().isoformat()}")
    log.info("=" * 50)

    try:
        cfg = load_config(CONFIG)
    except FileNotFoundError:
        cfg = {
            "groq_api_key":    os.getenv("GROQ_API_KEY", ""),
            "email_enabled":   os.getenv("EMAIL_ENABLED", "false").lower() == "true",
            "email_from":      os.getenv("EMAIL_FROM", ""),
            "email_to":        os.getenv("EMAIL_TO", ""),
            "smtp_host":       os.getenv("SMTP_HOST", "smtp-relay.brevo.com"),
            "smtp_port":       int(os.getenv("SMTP_PORT", "587")),
            "smtp_user":       os.getenv("SMTP_USER", ""),
            "smtp_pass":       os.getenv("SMTP_PASS", ""),
            "location":        os.getenv("LOCATION", "India"),
            "llm_batch_size":  5,
            "search_keywords": [
                "generative AI engineer fresher",
                "LLM engineer entry level",
                "AI engineer 2026 2027 batch",
                "machine learning engineer fresher",
                "software engineer AI india",
                "backend engineer AI startup india",
                "SDE fresher 2026 india",
            ],
        }

    log.info("Building crew...")
    crew = build_crew(cfg)

    log.info("Running crew...")
    result = crew.kickoff()

    log.info(f"Result: {result}")
    log.info("Run complete.")


if __name__ == "__main__":
    main()