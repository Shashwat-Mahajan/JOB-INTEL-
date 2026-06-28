"""
main.py — CrewAI pipeline entry point.

MIGRATION NOTE (v2.0 → v2.1)
──────────────────────────────
Only _default_config() changed:
  OLD: "groq_api_key": os.getenv("GROQ_API_KEY", "")
  NEW: "nvidia_nim_api_key": os.getenv("NVIDIA_NIM_API_KEY", "")

Everything else — keyword resolution, profile loading, crew building — unchanged.
"""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NO_PROXY"] = "huggingface.co,cdn-lfs.huggingface.co"

try:
    import tiktoken

    tiktoken.get_encoding("cl100k_base")
    print("tiktoken pre-loaded OK")
except Exception as e:
    print(f"tiktoken pre-load failed: {e}")

import logging
from datetime import date
from pathlib import Path

from utils import setup_logging, load_config, load_profile, get_search_keywords

BASE = Path(__file__).parent
LOGS = BASE / "logs" / "agent.log"
CONFIG = BASE / "config" / "config.json"

(BASE / "logs").mkdir(parents=True, exist_ok=True)
(BASE / "reports").mkdir(parents=True, exist_ok=True)

setup_logging(LOGS)
log = logging.getLogger(__name__)
log.info("Logger initialized: writing to %s", LOGS)

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
        # CHANGED: groq_api_key removed, two new keys added
        "nvidia_nim_api_key": os.getenv("NVIDIA_NIM_API_KEY", ""),
        "email_enabled": os.getenv("EMAIL_ENABLED", "false").lower() == "true",
        "email_from": os.getenv("EMAIL_FROM", ""),
        "email_to": os.getenv("EMAIL_TO", ""),
        "smtp_host": os.getenv("SMTP_HOST", "smtp-relay.brevo.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "smtp_user": os.getenv("SMTP_USER", ""),
        "smtp_pass": os.getenv("SMTP_PASS", ""),
        "location": os.getenv("LOCATION", "India"),
        "llm_batch_size": 10,
        "search_keywords": [],
        "include_internships": True,
    }


def main():
    log.info("=" * 50)
    log.info(f"Job Intel Agent — {date.today().isoformat()}")
    log.info("=" * 50)

    try:
        from crew import build_crew
        log.info("Crew imported")
    except Exception:
        log.exception("Failed to import crew")
        raise

    try:
        cfg = load_config(CONFIG)
        log.info("Crew imported")
    except FileNotFoundError:
        log.warning("config/config.json not found — using env vars and defaults")
        cfg = _default_config()

    # Allow env vars to override config.json (needed for GitHub Actions)
    if os.getenv("NVIDIA_NIM_API_KEY"):
        cfg["nvidia_nim_api_key"] = os.getenv("NVIDIA_NIM_API_KEY")

    # Validate required keys early — fail fast with a clear message
    if not cfg.get("nvidia_nim_api_key"):
        log.error(
            "nvidia_nim_api_key missing. " "Get a free key at: https://build.nvidia.com"
        )
        raise SystemExit(1)
    if not cfg.get("nvidia_nim_api_key"):
        log.error(
            "nvidia_nim_api_key missing. "
            "Get a free key at: https://build.nvidia.com — click any model → Get API Key."
        )
        raise SystemExit(1)

    profile = load_profile()
    if profile:
        cfg["profile"] = profile
        log.info(
            f"Profile loaded: {profile.get('name', 'unknown')} ({profile.get('graduation_batch', '')} batch)"
        )
    else:
        log.warning("config/profile.json not found — run: python setup_profile.py")

    if not cfg.get("search_keywords"):
        if profile:
            cfg["search_keywords"] = get_search_keywords(profile)
            log.info(
                f"Using {len(cfg['search_keywords'])} profile-driven search "
                f"keywords derived from target_roles in profile.json"
            )
        else:
            cfg["search_keywords"] = DEFAULT_KEYWORDS
            log.info(
                "No profile found and no config override — falling back to hardcoded DEFAULT_KEYWORDS"
            )

    if cfg.get("include_internships", True):
        intern_kw = [k for k in cfg["search_keywords"] if "intern" in k.lower()]
        if len(intern_kw) < 3:
            cfg["search_keywords"] = list(
                dict.fromkeys(
                    cfg["search_keywords"]
                    + [k for k in DEFAULT_KEYWORDS if "intern" in k.lower()]
                )
            )

    log.info("Building crew...")
    crew = build_crew(cfg)

    log.info("Running crew...")
    try:
        log.info("Starting crew.kickoff()")
        result = crew.kickoff()
    except Exception:
        log.exception("Crew crashed")
        raise
    log.info("Crew finished successfully")

    log.info(f"Result: {result}")
    log.info("Run complete.")


if __name__ == "__main__":
    main()
