"""Runtime-layer governance (Task 15): a per-request token / cost budget.

A RequestBudget is opened for each request (contextvar, so concurrent FastAPI requests
do not share counters). Every LLM call — mock or real — charges it. The input is
checked up front (oversized requests are rejected before any model call), and every
subsequent charge is checked too, so a request can never silently exceed its cap.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field

import config


class BudgetExceeded(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    """~4 characters per token (the usual English heuristic for BPE tokenisers)."""
    return max(1, (len(text) + 3) // 4)


@dataclass
class RequestBudget:
    max_input_tokens: int = config.MAX_INPUT_TOKENS_PER_REQUEST
    max_total_tokens: int = config.MAX_TOTAL_TOKENS_PER_REQUEST
    max_cost_usd: float = config.MAX_COST_PER_REQUEST_USD
    price_per_1k: float = config.PRICE_PER_1K_TOKENS_USD
    used_tokens: int = 0
    calls: list = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return round(self.used_tokens / 1000 * self.price_per_1k, 6)

    def check_input(self, user_text: str) -> int:
        n = estimate_tokens(user_text)
        if n > self.max_input_tokens:
            raise BudgetExceeded(
                f"Request rejected: input is ~{n} tokens, over the per-request input cap of "
                f"{self.max_input_tokens} tokens. No model call was made.")
        return n

    def charge(self, purpose: str, prompt: str, completion: str) -> None:
        n = estimate_tokens(prompt) + estimate_tokens(completion)
        projected = self.used_tokens + n
        projected_cost = projected / 1000 * self.price_per_1k
        if projected > self.max_total_tokens or projected_cost > self.max_cost_usd:
            raise BudgetExceeded(
                f"Request stopped: call '{purpose}' would bring usage to {projected} tokens "
                f"(${projected_cost:.5f}), over the cap of {self.max_total_tokens} tokens / "
                f"${self.max_cost_usd}.")
        self.used_tokens = projected
        self.calls.append((purpose, n))


_current: contextvars.ContextVar[RequestBudget | None] = contextvars.ContextVar("budget", default=None)


def current() -> RequestBudget | None:
    return _current.get()


@contextmanager
def request_budget(**overrides):
    b = RequestBudget(**overrides)
    token = _current.set(b)
    try:
        yield b
    finally:
        _current.reset(token)


def charge(purpose: str, prompt: str, completion: str) -> None:
    b = _current.get()
    if b is not None:
        b.charge(purpose, prompt, completion)
