"""
nim_client.py — LLM client using NVIDIA NIM for ALL inference.

Replaces gemini_client.py entirely.

- Scoring (Layers 2+3 in scorer.py)  → meta/llama-3.3-70b-instruct
- Verification (verify_jobs_tool)     → meta/llama-3.3-70b-instruct
- Profile extraction (setup_profile)  → meta/llama-3.3-70b-instruct
- Agent reasoning (crew.py _get_llm)  → meta/llama-3.1-8b-instruct  (unchanged)

NVIDIA NIM is fully OpenAI-compatible — uses openai SDK with a custom base_url.
Free tier: 1000 credits/month at build.nvidia.com (one key works for all models).

Install:
    pip install openai

Usage (imported by scorer.py, crew.py, setup_profile.py):
    from nim_client import call_nim, NIM_SCORING_MODEL
"""

import json
import logging
import time
from openai import OpenAI

log = logging.getLogger(__name__)

NIM_BASE_URL      = "https://integrate.api.nvidia.com/v1"
NIM_SCORING_MODEL = "meta/llama-3.3-70b-instruct"   # for scoring, verification, profile extraction
NIM_AGENT_MODEL   = "meta/llama-3.1-8b-instruct"    # for crew agent reasoning (already set in _get_llm)


def make_client(api_key: str) -> OpenAI:
    """Return a configured OpenAI client pointed at NVIDIA NIM."""
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
    Returns raw response text.

    For JSON outputs: include "Return ONLY valid JSON" in your system_prompt.
    NIM doesn't have response_mime_type like Gemini, but llama-3.3-70b
    follows JSON instructions reliably at temperature=0.1.
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
                wait = 30 * attempt   # 30s, 60s, 90s
                log.warning(
                    "NIM rate limit (attempt %d/%d) — waiting %ds: %s",
                    attempt, retries, wait, e,
                )
                time.sleep(wait)
            elif attempt < retries:
                log.warning("NIM error (attempt %d/%d): %s — retrying in 5s", attempt, retries, e)
                time.sleep(5)
            else:
                raise

    raise RuntimeError(f"NIM call failed after {retries} attempts")


def clean_json(raw: str) -> str:
    """Strip markdown fences if the model wraps output in ```json ... ```."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()
