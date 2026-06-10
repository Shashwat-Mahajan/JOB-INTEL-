"""
utils.py
Shared helpers: logging setup, seen-job dedup, file I/O.
"""

import json
import logging
from pathlib import Path


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_seen(path: Path) -> set:
    """Load set of already-seen job IDs from JSON file."""
    if path.exists():
        try:
            return set(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_seen(path: Path, seen: set) -> None:
    """Persist seen job IDs back to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sorted(list(seen)), indent=2),
        encoding="utf-8",
    )


def deduplicate(jobs: list[dict], seen: set) -> tuple[list[dict], set]:
    """
    Filter jobs to only unseen ones.
    Adds new IDs to seen set.
    Returns (fresh_jobs, updated_seen).
    """
    fresh = []
    for job in jobs:
        jid = job.get("job_id", "")
        if jid and jid not in seen:
            fresh.append(job)
            seen.add(jid)
    return fresh, seen


def load_config(path: Path) -> dict:
    """Load and return config.json as dict."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. "
            "Copy config/config.example.json to config/config.json and fill it in."
        )
    return json.loads(path.read_text(encoding="utf-8"))