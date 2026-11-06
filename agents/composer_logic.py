"""Deterministic composition rules used by the Response Composer under MOCK_LLM.

`render_lookup_sentences` is also used to build the review context for the Autogen team,
so a status sentence in the draft is checked against the exact record that was looked up.
"""
from __future__ import annotations

from agents.schemas import CrewResponse, LoanStatusResult
from rag.retrieval import IDK_ANSWER


def inr(amount: int) -> str:
    """Indian digit grouping: 1990000 -> 19,90,000."""
    s = str(int(amount))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def render_lookup_sentences(lk: dict) -> list[str]:
    if not lk.get("found"):
        return [f"No loan application was found with id {lk.get('record_id')}."]
    s = [f"Application {lk['record_id']} ({lk['category']}) is currently {lk['status']} "
         f"for a loan amount of Rs {inr(lk['loan_amount_inr'])}."]
    if lk["escalate"]:
        s.append(f"Its escalation score is {lk['escalation_score']}, at or above the threshold of "
                 f"{lk['escalation_threshold']}, so it should be escalated to a senior lending officer.")
    else:
        s.append(f"Its escalation score is {lk['escalation_score']}, below the threshold of "
                 f"{lk['escalation_threshold']}, so no escalation is needed.")
    return s


def compose(user_query: str, outputs: list[dict]) -> CrewResponse:
    rag = next((o for o in outputs if o.get("tool") == "rag_lookup"), None)
    lk = next((o for o in outputs if o.get("tool") == "check_loan_application_status"), None)
    parts, sources, tools_used = [], [], []
    grounded = True
    record = None
    if lk is not None:
        tools_used.append("check_loan_application_status")
        parts += render_lookup_sentences(lk)
        if lk.get("found"):
            record = LoanStatusResult(**{k: lk.get(k) for k in LoanStatusResult.model_fields})
        else:
            record = LoanStatusResult(record_id=lk.get("record_id", "?"), found=False)
    if rag is not None:
        tools_used.append("rag_lookup")
        grounded = bool(rag.get("grounded"))
        parts.append(rag["answer"] if grounded else IDK_ANSWER)
        sources = list(rag.get("sources", []))
    if lk is not None and rag is not None:
        intent = "policy+status"
    elif lk is not None:
        intent = "status"
    else:
        intent = "policy"
    return CrewResponse(answer=" ".join(parts) or IDK_ANSWER, intent=intent, grounded=grounded,
                        sources=sources, record_lookup=record,
                        escalate=bool(record and record.escalate), tools_used=tools_used)
