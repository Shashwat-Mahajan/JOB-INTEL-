"""
main.py — CrewAI pipeline entry point.

v2.2 changes:
  - DEFAULT_KEYWORDS removed — no more hardcoded AI/ML fallback keywords
  - _build_fallback_keywords() derives generic SDE/backend/intern keywords
  - include_internships block now derives intern keywords from profile target_roles
    instead of appending from hardcoded DEFAULT_KEYWORDS
  - Duplicate nvidia_nim_api_key validation block removed
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


def _build_fallback_keywords() -> list[str]:
    """
    Generic SDE/backend/fullstack keywords used ONLY when no profile.json exists.
    Not AI/ML specific — works for any engineering fresher.
    """
    return [
        "software engineer fresher india",
        "software engineer intern india",
        "SDE fresher 2027 india",
        "SDE intern india",
        "backend engineer fresher india",
        "backend developer intern india",
        "full stack developer fresher india",
        "full stack developer intern india",
        "java developer fresher india",
        "java developer intern india",
        "python developer fresher india",
        "python developer intern india",
        "web developer fresher india",
        "web developer intern india",
    ]


def _default_config() -> dict:
    return {
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
        log.info("Config loaded from config.json")
    except FileNotFoundError:
        log.warning("config/config.json not found — using env vars and defaults")
        cfg = _default_config()

    # Allow env vars to override config.json (needed for Render / GitHub Actions)
    if os.getenv("NVIDIA_NIM_API_KEY"):
        cfg["nvidia_nim_api_key"] = os.getenv("NVIDIA_NIM_API_KEY")
    if os.getenv("NVIDIA_NIM_API_KEY_2"):
        cfg["nvidia_nim_api_key_2"] = os.getenv("NVIDIA_NIM_API_KEY_2")

    # Validate required key early — fail fast with a clear message
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
            f"Profile loaded: {profile.get('name', 'unknown')} "
            f"({profile.get('graduation_batch', '')} batch)"
        )
    else:
        log.warning("config/profile.json not found — run: python setup_profile.py")

    # ── Build search keywords ─────────────────────────────────────────────────
    if not cfg.get("search_keywords"):
        if profile:
            cfg["search_keywords"] = get_search_keywords(profile)
            log.info(
                f"Using {len(cfg['search_keywords'])} profile-driven search keywords "
                f"derived from target_roles in profile.json"
            )
        else:
            cfg["search_keywords"] = _build_fallback_keywords()
            log.info(
                "No profile found — using generic SDE/backend fallback keywords. "
                "Run setup_profile.py for profile-specific keywords."
            )

    # ── Ensure intern keywords are profile-driven, not AI-hardcoded ──────────
    if cfg.get("include_internships", True):
        intern_kw = [k for k in cfg["search_keywords"] if "intern" in k.lower()]
        if len(intern_kw) < 3:
            # Derive intern variants from profile target_roles, not hardcoded list
            target_roles = profile.get("target_roles", []) if profile else []
            if target_roles:
                extra_intern = [
                    f"{role} intern"
                    for role in target_roles
                    if f"{role} intern".lower()
                    not in {k.lower() for k in cfg["search_keywords"]}
                ]
                cfg["search_keywords"] = list(
                    dict.fromkeys(cfg["search_keywords"] + extra_intern)
                )
                log.info(
                    f"Added {len(extra_intern)} intern keyword variants "
                    f"from profile target_roles"
                )
            else:
                # Fallback: add generic intern keywords
                generic_intern = [
                    k
                    for k in _build_fallback_keywords()
                    if "intern" in k.lower()
                    and k.lower() not in {kw.lower() for kw in cfg["search_keywords"]}
                ]
                cfg["search_keywords"] = list(
                    dict.fromkeys(cfg["search_keywords"] + generic_intern)
                )

    log.info(
        "Final search keywords (%d): %s",
        len(cfg["search_keywords"]),
        ", ".join(cfg["search_keywords"][:5])
        + ("..." if len(cfg["search_keywords"]) > 5 else ""),
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
