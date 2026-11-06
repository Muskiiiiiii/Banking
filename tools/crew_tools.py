"""CrewAI tool wrappers around the RAG core (Tasks 3–5) and the lookup tool (Task 6)."""
from __future__ import annotations

import contextvars
import json
from typing import ClassVar

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from rag.retrieval import default_generator
from tools.loan_status import check_loan_application_status

RETRIEVAL_ROLE = "Policy Retrieval Agent"
LOOKUP_ROLE = "Loan Lookup Agent"
COMPOSER_ROLE = "Response Composer"

# Per-request record of tool invocations (contextvar → safe under concurrent requests).
_tool_log: contextvars.ContextVar[list | None] = contextvars.ContextVar("tool_log", default=None)
# CrewAI may execute tools inside a copied context, so a ContextVar.set() made inside a tool
# would not be visible to the caller. We therefore store MUTABLE containers in the contextvars
# (created per request by start_tool_log) and mutate them from the tool.
_last_retrieval: contextvars.ContextVar[dict | None] = contextvars.ContextVar("last_retrieval", default=None)


def start_tool_log() -> list:
    log: list = []
    _tool_log.set(log)
    _last_retrieval.set({})
    return log


def last_retrieval() -> dict | None:
    box = _last_retrieval.get()
    return box.get("result") if box else None


def _record(name: str, args: dict) -> None:
    log = _tool_log.get()
    if log is not None:
        log.append({"tool": name, "args": args})


class RagLookupArgs(BaseModel):
    query: str = Field(..., description="The member's policy question, in plain language.")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = ("Search Cred's loan and banking policy knowledge base and return a grounded "
                        "answer with its source documents, or an 'I don't know' fallback.")
    args_schema: type[BaseModel] = RagLookupArgs

    def _run(self, query: str) -> str:
        _record(self.name, {"query": query})
        r = default_generator().answer(query)
        box = _last_retrieval.get()
        if box is not None:
            box["result"] = r
        return json.dumps({"tool": self.name, "answer": r["answer"], "grounded": r["grounded"],
                           "top1_similarity": r["top1_similarity"], "sources": r["sources"],
                           "cache_hit": r["cache_hit"]})


class LoanStatusArgs(BaseModel):
    record_id: str = Field(..., description="Loan application id in the form CRED-LA-0000.")


class LoanStatusTool(BaseTool):
    name: str = "check_loan_application_status"
    description: str = ("Look up one loan application by record id and return its status, amount "
                        "and escalation score.")
    args_schema: type[BaseModel] = LoanStatusArgs
    # Least-autonomy declaration enforced by agents/governance.py
    restricted_to_roles: ClassVar[frozenset[str]] = frozenset({LOOKUP_ROLE})

    def _run(self, record_id: str) -> str:
        _record(self.name, {"record_id": record_id})
        return json.dumps({"tool": self.name, **check_loan_application_status(record_id)})
