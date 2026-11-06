"""Thin text-completion layer used by grounded generation and the LLM-as-judge.

MOCK_LLM=1 (default): `mock_fn` produces a deterministic answer; no key, no network.
MOCK_LLM=0: the prompt is sent to an OpenAI-compatible chat model (optional path,
not used by the graded transcripts).

Every call — mock or real — is counted (LLM_STATS) and charged to the per-request
budget, so caching and the cost cap behave identically in both modes.
"""
from __future__ import annotations

from typing import Callable

import config
from agents import budget

LLM_STATS = {"calls": 0, "by_purpose": {}}


def complete(prompt: str, *, purpose: str, mock_fn: Callable[[], str]) -> str:
    LLM_STATS["calls"] += 1
    LLM_STATS["by_purpose"][purpose] = LLM_STATS["by_purpose"].get(purpose, 0) + 1
    budget.charge(purpose, prompt, "")  # pre-charge the prompt so oversized prompts stop early
    if config.MOCK_LLM:
        out = mock_fn()
    else:  # pragma: no cover - optional real path
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(
            model=config.REAL_LLM_MODEL, temperature=0,
            messages=[{"role": "user", "content": prompt}])
        out = resp.choices[0].message.content or ""
    budget.charge(purpose + ":completion", "", out)
    return out
