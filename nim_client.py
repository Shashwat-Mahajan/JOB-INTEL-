"""
nim_client.py — LLM client using NVIDIA NIM for ALL inference.

v2.4 changes — fix concurrent-thread key collision:
  - NEW: call_nim_rotated_pinned() — like call_nim_rotated(), but each
    concurrent caller pins to a specific key index instead of sharing the
    pool's mutable _index. Under v2.3, concurrent Layer-2 batches all called
    call_nim_rotated() which reads/rotates the SAME shared _pool._index.
    Two threads firing at once could land on the SAME "current" key
    simultaneously (doubling up requests on one key), and when that key
    failed and rotated, the OTHER thread's next attempt would often land on
    the newly-rotated key right behind it — both threads chasing the same
    key back and forth instead of using one key each in parallel. That is
    what produced the lockstep "key 1 fails -> rotate -> key 2 fails
    immediately -> rotate -> key 1 fails" pattern in the logs, even though
    two independent keys were configured.
    call_nim_rotated_pinned() fixes this by deriving each thread's starting
    key deterministically from a caller-supplied key_index (e.g. batch
    index), so batch 0 always starts on key 0, batch 1 always starts on
    key 1, etc. On failure it walks forward through the OTHER keys in the
    pool (by index, not via the shared rotating cursor) before giving up.
  - call_nim_batch_concurrent() updated to use the pinned variant internally
    for the same reason.
  - make_client(), call_nim(), call_nim_rotated() signatures/behavior
    UNCHANGED — still safe for single-key or non-concurrent call sites.

  IMPORTANT: to get any benefit from rotation/concurrency you must actually
  set a second key:
      NVIDIA_NIM_API_KEY=nvapi-...      (primary)
      NVIDIA_NIM_API_KEY_2=nvapi-...    (secondary — get a free second key
                                          from a second NVIDIA account/org,
                                          NIM keys are free-tier per account)

- Scoring (Layers 2+3 in scorer.py)  → meta/llama-3.3-70b-instruct
- Verification (verify_jobs_tool)     → meta/llama-3.3-70b-instruct
- Profile extraction (setup_profile)  → meta/llama-3.3-70b-instruct
- Agent reasoning (crew.py _get_llm)  → meta/llama-3.3-70b-instruct

Install:
    pip install openai

Usage:
    from nim_client import call_nim, call_nim_rotated, call_nim_rotated_pinned, call_nim_batch_concurrent, NIM_SCORING_MODEL
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

log = logging.getLogger(__name__)

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_SCORING_MODEL = "meta/llama-3.3-70b-instruct"
NIM_AGENT_MODEL = "meta/llama-3.1-8b-instruct"

# Per-request timeout (seconds). Without this, a stalled connection can
# hang far longer than your own retry logic expects before the SDK
# even raises an exception for you to catch.
NIM_REQUEST_TIMEOUT = 30.0

# The openai SDK retries failed requests internally by default (max_retries=2),
# with its own backoff, BEFORE the exception ever reaches our try/except below.
# Setting max_retries=0 here makes our own retry/rotation logic the only
# thing in control of timing.
NIM_SDK_MAX_RETRIES = 0

# Connection-error retry backoff base (seconds).
CONN_ERROR_BASE_WAIT = 2.0

# Rate-limit retry backoff base (seconds).
RATE_LIMIT_BASE_WAIT = 8.0


# ── Dual-key pool ─────────────────────────────────────────────────────────────


class _KeyPool:
    """
    Manages a pool of NVIDIA NIM API keys with round-robin rotation.
    On rate limit OR connection error from one key, tries the next.

    Thread-safety note: with concurrent batch scoring (multiple threads
    calling call_nim_rotated() at once), _index is shared mutable state.
    A lock is used around all read-modify-write access to avoid two threads
    racing on rotate()/current() and producing confusing interleaved log
    output.

    IMPORTANT: this shared rotating cursor (current()/rotate()) is fine for
    SEQUENTIAL callers, but is NOT a good fit for concurrent callers that
    each want their own dedicated key — concurrent threads can converge on
    the same "current" key at once. For concurrent dispatch, use key_at()
    with a caller-owned deterministic offset instead (see
    call_nim_rotated_pinned() below), which doesn't touch _index at all.
    """

    def __init__(self):
        self._keys: list[str] = []
        self._index: int = 0
        self._loaded: bool = False
        self._lock = threading.Lock()

    def _load(self):
        if self._loaded:
            return
        k1 = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
        k2 = os.getenv("NVIDIA_NIM_API_KEY_2", "").strip()

        if k1:
            self._keys.append(k1)
        if k2 and k2 != k1:
            self._keys.append(k2)

        if not self._keys:
            log.warning("No NVIDIA NIM API keys found in environment")
        elif len(self._keys) == 1:
            log.info(
                "NIM key pool: 1 key loaded (single-key mode — no rotation/concurrency benefit)"
            )
        else:
            log.info(
                "NIM key pool: %d keys loaded (rotation + concurrency active)",
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
        with self._lock:
            if not self._keys:
                return ""
            return self._keys[self._index % len(self._keys)]

    def key_at(self, offset: int) -> str:
        """Get a key at a specific offset, without mutating shared state.
        Used by concurrent/pinned callers so parallel threads spread across
        keys deterministically instead of racing on self._index."""
        self._load()
        if not self._keys:
            return ""
        return self._keys[offset % len(self._keys)]

    def rotate(self):
        with self._lock:
            if len(self._keys) > 1:
                self._index = (self._index + 1) % len(self._keys)
                idx_display = self._index + 1
                size = len(self._keys)
        if len(self._keys) > 1:
            log.info("NIM key pool: rotated to key %d/%d", idx_display, size)

    def size(self) -> int:
        self._load()
        return len(self._keys)

    def current_index(self) -> int:
        """Safe read of the current index for logging — avoids touching _index directly."""
        with self._lock:
            return self._index % len(self._keys) if self._keys else 0


_pool = _KeyPool()


# ── Per-key rate limiter ──────────────────────────────────────────────────────
# NVIDIA NIM's free tier caps each key at 40 requests/minute. Key rotation on
# failure doesn't help if you're simply EXCEEDING that cap — you'll just hit
# the limit on both keys in turn. This is a simple per-key token-bucket style
# limiter: before any request goes out on a given key, we check how many
# requests that key has made in the trailing 60s window and sleep if needed
# to stay under the cap. Set conservatively below 40 to leave headroom for
# NVIDIA-side jitter/clock skew.
NIM_MAX_RPM_PER_KEY = 35


class _RateLimiter:
    def __init__(self, max_rpm: int):
        self._max_rpm = max_rpm
        self._timestamps: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def wait_for_slot(self, key: str):
        while True:
            with self._lock:
                now = time.time()
                window_start = now - 60.0
                ts = self._timestamps.setdefault(key, [])
                # drop anything older than the 60s window
                ts[:] = [t for t in ts if t > window_start]

                if len(ts) < self._max_rpm:
                    ts.append(now)
                    return

                # need to wait until the oldest request in the window expires
                sleep_for = ts[0] + 60.0 - now

            if sleep_for > 0:
                log.info(
                    "NIM rate limiter: key ...%s at %d/%d req/min — waiting %.1fs",
                    key[-4:] if len(key) >= 4 else key,
                    self._max_rpm,
                    self._max_rpm,
                    sleep_for,
                )
                time.sleep(sleep_for)
            # loop back and re-check — another thread may have also been
            # waiting on the same key


_rate_limiter = _RateLimiter(NIM_MAX_RPM_PER_KEY)


# ── Groq fallback ──────────────────────────────────────────────────────────────
# When NIM is saturated (both keys rate-limited or erroring), fall back to
# Groq's openai/gpt-oss-120b — OpenAI's open-weight reasoning model, free
# tier 30 RPM / 1,000 RPD. Reasoning quality is meaningfully better than
# vanilla Llama for judge/scoring tasks, and it's a fully independent quota
# pool from NIM, so it absorbs overflow instead of compounding NIM's limits.
#
# Note: Groq deprecated deepseek-r1-distill-llama-70b (Sep 2025) and, as of
# June 17 2026, llama-3.3-70b-versatile / llama-3.1-8b-instant too — Groq's
# own guidance is to migrate to openai/gpt-oss-120b (or qwen/qwen3.6-27b),
# so that's what's used here rather than the older deprecated model names.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_FALLBACK_MODEL = "openai/gpt-oss-120b"
GROQ_REQUEST_TIMEOUT = 30.0
GROQ_MAX_RPM = 28  # stay under Groq's 30 RPM free-tier cap, headroom for jitter

_groq_rate_limiter = _RateLimiter(GROQ_MAX_RPM)


def _get_groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def call_groq_fallback(
    system_prompt: str,
    user_content: str,
    model: str = GROQ_FALLBACK_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """
    Single call to Groq as a fallback when NIM is unavailable/saturated.
    Raises RuntimeError if GROQ_API_KEY isn't set, or the underlying
    exception if the Groq call itself fails (no further fallback beyond
    this — caller decides what to do next, e.g. give up on this batch).
    """
    groq_key = _get_groq_key()
    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY not set — cannot use Groq fallback. "
            "Set the GROQ_API_KEY environment variable to enable it."
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    client = OpenAI(
        api_key=groq_key,
        base_url=GROQ_BASE_URL,
        timeout=GROQ_REQUEST_TIMEOUT,
        max_retries=0,
    )

    _groq_rate_limiter.wait_for_slot(groq_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ── Groq key pool (3 keys, for primary scoring + verification) ────────────────
# Used directly for Layer 2 + Layer 3 now instead of NIM. Same pinned-rotation
# pattern as the NIM key pool: each concurrent caller gets a deterministic
# starting key by index, so concurrent batches don't converge on the same key.

GROQ_MAX_RPM_PER_KEY = 28  # stay under Groq's 30 RPM free-tier cap per key


class _GroqKeyPool:
    def __init__(self):
        self._keys: list[str] = []
        self._loaded: bool = False

    def add_key(self, api_key: str):
        key = (api_key or "").strip()
        if key and key not in self._keys:
            self._keys.append(key)
            self._loaded = True

    def key_at(self, offset: int) -> str:
        if not self._keys:
            return ""
        return self._keys[offset % len(self._keys)]

    def size(self) -> int:
        return len(self._keys)


_groq_pool = _GroqKeyPool()
_groq_rate_limiters: dict[str, _RateLimiter] = {}


def _groq_limiter_for(key: str) -> _RateLimiter:
    # one limiter per key so each key's 28rpm budget is tracked independently
    if key not in _groq_rate_limiters:
        _groq_rate_limiters[key] = _RateLimiter(GROQ_MAX_RPM_PER_KEY)
    return _groq_rate_limiters[key]


def register_groq_keys_from_config(cfg: dict):
    """
    Register up to 3 Groq keys from config.json fields:
        "GROQ_API_KEY 1", "GROQ_API_KEY 2", "GROQ_API_KEY 3"
    Call once at startup, same place register_keys_from_config() is called.
    """
    added = 0
    for field in ("GROQ_API_KEY 1", "GROQ_API_KEY 2", "GROQ_API_KEY 3"):
        k = cfg.get(field, "")
        if k:
            _groq_pool.add_key(k)
            added += 1
    if added:
        log.info(
            "Groq key pool: registered %d key(s) from config.json (pool size=%d)",
            added,
            _groq_pool.size(),
        )
    else:
        log.warning(
            "register_groq_keys_from_config: no Groq keys found under "
            "'GROQ_API_KEY 1/2/3' in config.json"
        )


def groq_pool_size() -> int:
    return _groq_pool.size()


def call_groq_rotated_pinned(
    system_prompt: str,
    user_content: str,
    key_index: int,
    model: str = GROQ_FALLBACK_MODEL,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """
    Groq call for CONCURRENT callers — pins each call to a deterministic key
    by index (same pattern as call_nim_rotated_pinned). On failure, walks
    forward through the other keys in the pool before giving up.
    """
    size = max(1, _groq_pool.size())
    if _groq_pool.size() == 0:
        raise RuntimeError(
            "No Groq API keys registered — call register_groq_keys_from_config(cfg) at startup"
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    last_err = None

    for attempt in range(1, retries + 1):
        offset = (key_index + attempt - 1) % size
        current_key = _groq_pool.key_at(offset)
        if not current_key:
            raise RuntimeError("No Groq API keys available")

        client = OpenAI(
            api_key=current_key,
            base_url=GROQ_BASE_URL,
            timeout=GROQ_REQUEST_TIMEOUT,
            max_retries=0,
        )

        try:
            _groq_limiter_for(current_key).wait_for_slot(current_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            err = str(e).lower()
            retryable = _is_rate_limit(err) or _is_retryable_connection_error(err)

            if retryable and size > 1:
                next_offset = (key_index + attempt) % size
                log.warning(
                    "Groq error on pinned key %d/%d (key_index=%d, %s) — trying key %d/%d next",
                    offset + 1,
                    size,
                    key_index,
                    e,
                    next_offset + 1,
                    size,
                )
                continue
            elif _is_rate_limit(err):
                wait = RATE_LIMIT_BASE_WAIT * attempt
                log.warning(
                    "Groq rate limit on pinned key (key_index=%d, attempt %d/%d) — waiting %.0fs: %s",
                    key_index,
                    attempt,
                    retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            elif attempt < retries:
                wait = CONN_ERROR_BASE_WAIT * attempt
                log.warning(
                    "Groq error on pinned key (key_index=%d, attempt %d/%d): %s — retrying in %.0fs",
                    key_index,
                    attempt,
                    retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"Groq pinned call (key_index={key_index}) failed after {retries} attempts — last error: {last_err}"
    )


def call_llm_with_fallback(
    api_key: str,
    system_prompt: str,
    user_content: str,
    key_index: int = 0,
    model: str = NIM_SCORING_MODEL,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    """
    Tries NIM (pinned by key_index) first; on failure, falls back to Groq's
    gpt-oss-120b if GROQ_API_KEY is configured.

    Returns (response_text, source) where source is "nim" or "groq_fallback"
    — useful for logging/debugging which provider actually answered a
    given batch.

    Raises the original NIM error if Groq fallback isn't configured or
    also fails, so callers' existing retry/error-handling logic upstream
    still works as before.
    """
    try:
        out = call_nim_rotated_pinned(
            api_key,
            system_prompt,
            user_content,
            key_index=key_index,
            model=model,
            retries=retries,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return out, "nim"
    except Exception as nim_err:
        if not _get_groq_key():
            # No fallback configured — surface the original NIM error
            raise

        log.warning(
            "NIM exhausted (key_index=%d) — falling back to Groq (%s): %s",
            key_index,
            GROQ_FALLBACK_MODEL,
            nim_err,
        )
        try:
            out = call_groq_fallback(
                system_prompt,
                user_content,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return out, "groq_fallback"
        except Exception as groq_err:
            log.error(
                "Groq fallback also failed (key_index=%d): %s — "
                "raising original NIM error",
                key_index,
                groq_err,
            )
            raise nim_err


def _is_rate_limit(err_str: str) -> bool:
    return "429" in err_str or "rate" in err_str or "quota" in err_str


def _is_retryable_connection_error(err_str: str) -> bool:
    return any(
        s in err_str
        for s in (
            "connection error",
            "timeout",
            "timed out",
            "connection reset",
            "eof occurred",
        )
    )


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
        timeout=NIM_REQUEST_TIMEOUT,
        max_retries=NIM_SDK_MAX_RETRIES,
    )


def register_keys_from_config(cfg: dict):
    """
    Register one or more NIM keys stored in config.json instead of env vars.

    Supports either of these shapes in config.json:
        {"nvidia_nim_api_key": "nvapi-...", "nvidia_nim_api_key_2": "nvapi-..."}
    or:
        {"nvidia_nim_api_keys": ["nvapi-...", "nvapi-..."]}

    Call this once, early — e.g. right after config.json is loaded in
    __main__.py / crew.py — before any scoring/verification calls happen.
    Safe to call multiple times; duplicates are ignored by add_key().
    """
    added = 0

    keys_list = cfg.get("nvidia_nim_api_keys")
    if isinstance(keys_list, list):
        for k in keys_list:
            if k:
                _pool.add_key(k)
                added += 1

    k1 = cfg.get("nvidia_nim_api_key")
    if k1:
        _pool.add_key(k1)
        added += 1

    k2 = cfg.get("nvidia_nim_api_key_2")
    if k2:
        _pool.add_key(k2)
        added += 1

    log.info(
        "register_keys_from_config: cfg had nvidia_nim_api_key=%s nvidia_nim_api_key_2=%s nvidia_nim_api_keys=%s",
        "set" if k1 else "MISSING/EMPTY",
        "set" if k2 else "MISSING/EMPTY",
        f"{len(keys_list)} items" if isinstance(keys_list, list) else "MISSING",
    )

    if added:
        log.info(
            "register_keys_from_config: registered %d key(s) from config.json (pool size=%d)",
            added,
            _pool.size(),
        )
    else:
        log.warning("register_keys_from_config: no NIM keys found in config.json")


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
            if _is_rate_limit(err):
                wait = RATE_LIMIT_BASE_WAIT * attempt
                log.warning(
                    "NIM rate limit (attempt %d/%d) — waiting %.0fs: %s",
                    attempt,
                    retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            elif attempt < retries:
                wait = CONN_ERROR_BASE_WAIT * attempt
                log.warning(
                    "NIM error (attempt %d/%d): %s — retrying in %.0fs",
                    attempt,
                    retries,
                    e,
                    wait,
                )
                time.sleep(wait)
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
    NIM call with dual-key rotation using the pool's SHARED cursor.

    Good for sequential/single-threaded call sites. NOT recommended for
    concurrent dispatch (multiple threads calling this at once) — use
    call_nim_rotated_pinned() instead, since this function's key choice
    depends on shared mutable state that concurrent callers can race on
    and converge onto the same key. Kept unchanged for backward
    compatibility with existing non-concurrent call sites.
    """
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

        client = OpenAI(
            api_key=current_key,
            base_url=NIM_BASE_URL,
            timeout=NIM_REQUEST_TIMEOUT,
            max_retries=NIM_SDK_MAX_RETRIES,
        )

        try:
            _rate_limiter.wait_for_slot(current_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            _pool.rotate()
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            err = str(e).lower()
            retryable = _is_rate_limit(err) or _is_retryable_connection_error(err)

            if retryable and _pool.size() > 1:
                log.warning(
                    "NIM error on key %d/%d (%s) — switching to next key immediately",
                    _pool.current_index() + 1,
                    _pool.size(),
                    e,
                )
                _pool.rotate()
            elif _is_rate_limit(err):
                wait = RATE_LIMIT_BASE_WAIT * attempt
                log.warning(
                    "NIM rate limit (attempt %d/%d, single key) — waiting %.0fs: %s",
                    attempt,
                    retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            elif attempt < retries:
                wait = CONN_ERROR_BASE_WAIT * attempt
                log.warning(
                    "NIM error (attempt %d/%d): %s — retrying in %.0fs",
                    attempt,
                    retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"NIM rotated call failed after {retries} attempts — last error: {last_err}"
    )


def call_nim_rotated_pinned(
    api_key: str,
    system_prompt: str,
    user_content: str,
    key_index: int,
    model: str = NIM_SCORING_MODEL,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """
    NIM call for CONCURRENT callers — pins each call to a deterministic key
    by index instead of sharing the pool's rotating cursor.

    Why this exists: call_nim_rotated() reads/rotates _pool._index, which
    is shared across all threads. When N batches fire concurrently, they
    can all read the SAME "current" key at once (piling 2+ requests onto
    one key while the other sits idle), and after a failure-triggered
    rotate(), the next thread to check current() often lands on the same
    key the failing thread just rotated to — so concurrent threads chase
    each other onto the same key instead of using separate keys in
    parallel. This was the root cause of the lockstep "key 1 fails ->
    rotate -> key 2 fails immediately -> rotate -> key 1 fails" pattern
    seen under concurrent Layer-2 batch dispatch.

    key_index: caller-supplied identity for this concurrent unit of work
               (e.g. batch index). Combined with the pool size, this
               determines which key is tried first — batch 0 always starts
               on key 0, batch 1 always starts on key 1, etc. — so
               concurrent threads spread across keys instead of converging.
               On failure, retries walk forward through the OTHER keys in
               the pool (by index, not via the shared cursor) before
               giving up.

    Does NOT call _pool.rotate() — pinned calls never touch the shared
    cursor, so they can't disturb call_nim_rotated() callers elsewhere
    (e.g. Layer 3's sequential verifier call) and vice versa.
    """
    _pool.add_key(api_key)
    size = max(1, _pool.size())

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    last_err = None

    for attempt in range(1, retries + 1):
        # Own key first (attempt 1), then walk forward through the others.
        offset = (key_index + attempt - 1) % size
        current_key = _pool.key_at(offset)
        if not current_key:
            raise RuntimeError("No NVIDIA NIM API keys available")

        client = OpenAI(
            api_key=current_key,
            base_url=NIM_BASE_URL,
            timeout=NIM_REQUEST_TIMEOUT,
            max_retries=NIM_SDK_MAX_RETRIES,
        )

        try:
            _rate_limiter.wait_for_slot(current_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            err = str(e).lower()
            retryable = _is_rate_limit(err) or _is_retryable_connection_error(err)

            if retryable and size > 1:
                next_offset = (key_index + attempt) % size
                log.warning(
                    "NIM error on pinned key %d/%d (batch key_index=%d, %s) — "
                    "trying key %d/%d next",
                    offset + 1,
                    size,
                    key_index,
                    e,
                    next_offset + 1,
                    size,
                )
                # No sleep — go straight to the next (idle) key, since the
                # whole point of pinning is that other keys aren't being
                # hammered by other threads right now.
                continue
            elif _is_rate_limit(err):
                wait = RATE_LIMIT_BASE_WAIT * attempt
                log.warning(
                    "NIM rate limit on pinned key (batch key_index=%d, attempt %d/%d, "
                    "single key) — waiting %.0fs: %s",
                    key_index,
                    attempt,
                    retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            elif attempt < retries:
                wait = CONN_ERROR_BASE_WAIT * attempt
                log.warning(
                    "NIM error on pinned key (batch key_index=%d, attempt %d/%d): %s "
                    "— retrying in %.0fs",
                    key_index,
                    attempt,
                    retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"NIM pinned call (key_index={key_index}) failed after {retries} attempts "
        f"— last error: {last_err}"
    )


def call_nim_batch_concurrent(
    requests: list[tuple[str, str]],
    model: str = NIM_SCORING_MODEL,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_workers: int | None = None,
) -> list[str]:
    """
    Run multiple independent NIM calls concurrently, spread across the key pool.

    requests: list of (system_prompt, user_content) tuples — e.g. your
              5 scoring batches. Order of results matches order of input.

    Uses call_nim_rotated_pinned() internally (not call_nim_rotated()) so
    each request is pinned to its own key by index — concurrent threads
    don't converge on the same "current" key the way they could with the
    shared-cursor rotation.

    max_workers: defaults to pool size (so each key gets one in-flight
                 request at a time). With 1 key loaded, this effectively
                 runs sequentially.
    """
    if max_workers is None:
        max_workers = max(1, _pool.size())

    results: list = [None] * len(requests)

    def _worker(idx: int, system_prompt: str, user_content: str) -> tuple[int, str]:
        # key_index=idx pins this request to key (idx % pool_size) first.
        out = call_nim_rotated_pinned(
            api_key=_pool.key_at(idx),
            system_prompt=system_prompt,
            user_content=user_content,
            key_index=idx,
            model=model,
            retries=retries,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return idx, out

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker, i, sys_p, user_c): i
            for i, (sys_p, user_c) in enumerate(requests)
        }
        for future in as_completed(futures):
            idx, output = future.result()  # exceptions propagate naturally
            results[idx] = output

    return results


def clean_json(raw: str) -> str:
    """Strip markdown fences if the model wraps output in ```json ... ```."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def nim_pool_size() -> int:
    """
    Public helper so callers (e.g. scorer.py) can size their own concurrency
    (thread pool workers, batch fan-out, etc.) based on how many NIM keys
    are actually registered, without reaching into _pool directly.
    """
    return _pool.size()
