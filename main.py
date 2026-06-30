"""
main.py — CrewAI pipeline entry point.

v2.4 changes:
  - Fallback keywords removed entirely. No more _build_fallback_keywords().
    A profile.json is now REQUIRED — if it's missing, the run fails fast
    with a clear message instead of silently falling back to generic
    SDE/backend keywords. Run setup_profile.py first if you see this error.
  - include_internships block now ONLY derives intern keywords from
    profile target_roles. No generic fallback intern keywords either.

v2.3 changes:
  - Registers nvidia_nim_api_key / nvidia_nim_api_key_2 from config.json (or env)
    into nim_client's key pool via register_keys_from_config(), BEFORE build_crew()
    is called. This is what actually activates dual-key rotation/concurrency in
    nim_client.py.

v2.2 changes:
  - DEFAULT_KEYWORDS removed — no more hardcoded AI/ML fallback keywords
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


def _default_config() -> dict:
    return {
        "nvidia_nim_api_key": os.getenv("NVIDIA_NIM_API_KEY", ""),
        "nvidia_nim_api_key_2": os.getenv("NVIDIA_NIM_API_KEY_2", ""),
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

    # ── Register all available NIM keys into the shared key pool ─────────────
    # This is what activates rotation + concurrent batch scoring in nim_client.py.
    # Without this call, cfg["nvidia_nim_api_key_2"] just sits in the dict unused.
    try:
        from nim_client import register_keys_from_config

        register_keys_from_config(cfg)

        from nim_client import register_groq_keys_from_config
        register_groq_keys_from_config(cfg)
    except Exception:
        log.exception(
            "Failed to register NIM keys into key pool — continuing with single-key mode"
        )

    # ── Profile is now REQUIRED — no more generic fallback keywords ──────────
    profile = load_profile()
    if not profile:
        log.error(
            "config/profile.json not found. A profile is now required — "
            "there is no generic keyword fallback anymore. "
            "Run: python setup_profile.py"
        )
        raise SystemExit(1)

    cfg["profile"] = profile
    log.info(
        f"Profile loaded: {profile.get('name', 'unknown')} "
        f"({profile.get('graduation_batch', '')} batch)"
    )

    # ── Build search keywords — always profile-driven, no fallback ───────────
    if not cfg.get("search_keywords"):
        cfg["search_keywords"] = get_search_keywords(profile)
        log.info(
            f"Using {len(cfg['search_keywords'])} profile-driven search keywords "
            f"derived from target_roles in profile.json"
        )

    if not cfg["search_keywords"]:
        log.error(
            "search_keywords is empty after deriving from profile.json. "
            "Check that profile.json has a non-empty 'target_roles' list."
        )
        raise SystemExit(1)

    # ── Ensure intern keywords exist when internships are wanted ─────────────
    # Derived ONLY from profile target_roles now — no generic fallback list.
    if cfg.get("include_internships", True):
        intern_kw = [k for k in cfg["search_keywords"] if "intern" in k.lower()]
        if len(intern_kw) < 3:
            target_roles = profile.get("target_roles", [])
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
    except Exception as e:
        # v2.4: one retry for transient connection errors that crash the whole
        # run on the very first LLM call (seen in practice: NIM connection
        # drops before any retry logic inside LiteLLM/CrewAI's LLM object
        # kicks in). A genuine config/auth problem will fail again identically
        # on retry and surface the real traceback.
        err_str = str(e).lower()
        transient = any(
            s in err_str for s in ("connection error", "timeout", "timed out")
        )
        if transient:
            log.warning(
                "Crew crashed on a transient connection error — retrying once: %s", e
            )
            try:
                crew = build_crew(cfg)
                result = crew.kickoff()
            except Exception:
                log.exception("Crew crashed again on retry — giving up")
                raise
        else:
            log.exception("Crew crashed")
            raise

    log.info("Crew finished successfully")
    log.info(f"Result: {result}")
    log.info("Run complete.")


if __name__ == "__main__":
    main()
