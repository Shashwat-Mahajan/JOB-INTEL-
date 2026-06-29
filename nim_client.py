"""
nim_client.py — LLM client using NVIDIA NIM for ALL inference.

v2.2 changes — dual API key rotation:
  - Accepts two NVIDIA NIM API keys (primary + secondary)
  - Rotates between keys per call to halve rate limit pressure
  - On 429 from one key, immediately switches to the other
  - Keys loaded from env: NVIDIA_NIM_API_KEY (primary) + NVIDIA_NIM_API_KEY_2 (secondary)
  - Falls back gracefully to single-key mode if only one key is set
  - make_client() unchanged — still accepts a single key for crew.py/_get_llm()
  - call_nim() unchanged signature — rotation is internal to this module
  - New: call_nim_rotated() — use this in scorer.py and setup_profile.py
    for automatic dual-key rotation without changing call sites

- Scoring (Layers 2+3 in scorer.py)  → meta/llama-3.3-70b-instruct
- Verification (verify_jobs_tool)     → meta/llama-3.3-70b-instruct
- Profile extraction (setup_profile)  → meta/llama-3.3-70b-instruct
- Agent reasoning (crew.py _get_llm)  → meta/llama-3.3-70b-instruct

Install:
    pip install openai

Usage:
    from nim_client import call_nim, call_nim_rotated, NIM_SCORING_MODEL
"""

import json
import logging
import os
import time
from openai import OpenAI

log = logging.getLogger(__name__)

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_SCORING_MODEL = "meta/llama-3.3-70b-instruct"
NIM_AGENT_MODEL = "meta/llama-3.1-8b-instruct"


# ── Dual-key pool ─────────────────────────────────────────────────────────────


class _KeyPool:
    """
    Manages a pool of NVIDIA NIM API keys with round-robin rotation.
    On rate limit from one key, immediately tries the next.
    Thread-safe for single-threaded use (CrewAI sequential pipeline).
    """

    def __init__(self):
        self._keys: list[str] = []
        self._index: int = 0
        self._loaded: bool = False

    def _load(self):
        if self._loaded:
            return
        # Primary key — always required
        k1 = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
        # Secondary key — optional
        k2 = os.getenv("NVIDIA_NIM_API_KEY_2", "").strip()

        if k1:
            self._keys.append(k1)
        if k2 and k2 != k1:
            self._keys.append(k2)

        if not self._keys:
            log.warning("No NVIDIA NIM API keys found in environment")
        elif len(self._keys) == 1:
            log.info("NIM key pool: 1 key loaded (single-key mode)")
        else:
            log.info(
                "NIM key pool: %d keys loaded (dual-key rotation active)",
                len(self._keys),
            )

        self._loaded = True

    def add_key(self, api_key: str):
        """Add a key from config.json (called by scorer/setup_profile with cfg key)."""
        self._load()
        key = api_key.strip()
        if key and key not in self._keys:
            self._keys.insert(0, key)  # config key takes priority
            log.info("NIM key pool: added config key (pool size=%d)", len(self._keys))

    def current(self) -> str:
        self._load()
        if not self._keys:
            return ""
        return self._keys[self._index % len(self._keys)]

    def rotate(self):
        """Move to next key."""
        if len(self._keys) > 1:
            self._index = (self._index + 1) % len(self._keys)
            log.info(
                "NIM key pool: rotated to key %d/%d", self._index + 1, len(self._keys)
            )

    def size(self) -> int:
        self._load()
        return len(self._keys)


_pool = _KeyPool()


# ── Public helpers ────────────────────────────────────────────────────────────


def make_client(api_key: str) -> OpenAI:
    """
    Return a configured OpenAI client pointed at NVIDIA NIM.
    Also registers the key in the pool for rotated calls.
    Unchanged signature — crew.py and setup_profile.py call this as before.
    """
    _pool.add_key(api_key)
    return OpenAI(
        api_key=api_key,
        base_url=NIM_BASE_URL,
    )


def call_nim(
    client: OpenAI,
    system_prompt: str,
    user_content: str,
    model: str = NIM_SCORING_MODEL,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """
    Single NIM call with retry on rate limit / transient errors.
    Uses the provided client (single-key mode).
    Unchanged signature — existing callers work without modification.

    For batch scoring in scorer.py, prefer call_nim_rotated() which
    automatically switches keys on rate limit.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err or "quota" in err:
                wait = 30 * attempt
                log.warning(
                    "NIM rate limit (attempt %d/%d) — waiting %ds: %s",
                    attempt,
                    retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            elif attempt < retries:
                log.warning(
                    "NIM error (attempt %d/%d): %s — retrying in 5s",
                    attempt,
                    retries,
                    e,
                )
                time.sleep(5)
            else:
                raise

    raise RuntimeError(f"NIM call failed after {retries} attempts")


def call_nim_rotated(
    api_key: str,
    system_prompt: str,
    user_content: str,
    model: str = NIM_SCORING_MODEL,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """
    NIM call with dual-key rotation.

    Differences from call_nim():
    - Takes api_key string instead of a pre-built client
    - On 429, immediately rotates to the next key and retries
      instead of waiting 30s on the same key
    - Falls back to single-key wait if only one key is available

    Use this in scorer.py's _nim_call() for batch scoring
    to distribute load across both keys.
    """
    # Ensure the passed key is in the pool
    _pool.add_key(api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    last_err = None

    for attempt in range(1, retries + 1):
        current_key = _pool.current()
        if not current_key:
            raise RuntimeError("No NVIDIA NIM API keys available")

        client = OpenAI(api_key=current_key, base_url=NIM_BASE_URL)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Success — rotate to next key for the following call (round-robin)
            _pool.rotate()
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            err = str(e).lower()

            if "429" in err or "rate" in err or "quota" in err:
                if _pool.size() > 1:
                    # Switch to other key immediately — no sleep needed
                    log.warning(
                        "NIM rate limit on key %d/%d — switching to next key immediately",
                        (_pool._index % _pool.size()) + 1,
                        _pool.size(),
                    )
                    _pool.rotate()
                else:
                    # Single key — fall back to timed wait
                    wait = 30 * attempt
                    log.warning(
                        "NIM rate limit (attempt %d/%d, single key) — waiting %ds: %s",
                        attempt,
                        retries,
                        wait,
                        e,
                    )
                    time.sleep(wait)

            elif attempt < retries:
                log.warning(
                    "NIM error (attempt %d/%d): %s — retrying in 5s",
                    attempt,
                    retries,
                    e,
                )
                time.sleep(5)
            else:
                raise

    raise RuntimeError(
        f"NIM rotated call failed after {retries} attempts — last error: {last_err}"
    )


def clean_json(raw: str) -> str:
    """Strip markdown fences if the model wraps output in ```json ... ```."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()
