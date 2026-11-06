"""Pydantic structured-output schemas (Task 9 and Task 14)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LoanStatusResult(BaseModel):
    record_id: str
    found: bool
    category: Optional[str] = None
    status: Optional[str] = None
    loan_amount_inr: Optional[int] = None
    escalation_score: Optional[float] = Field(default=None, ge=0, le=1)
    escalate: Optional[bool] = None


class CrewResponse(BaseModel):
    """Every crew response (and every final API answer) must validate against this."""
    answer: str = Field(min_length=1)
    intent: Literal["policy", "status", "policy+status", "refused", "clarify"]
    grounded: bool
    sources: list[str] = Field(default_factory=list)
    record_lookup: Optional[LoanStatusResult] = None
    escalate: bool = False
    tools_used: list[str] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)


class ReviewVerdict(BaseModel):
    """Structured output of the Autogen Final-Editor agent."""
    approved: bool
    final_answer: str
    reason: str
