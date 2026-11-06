"""End-to-end request pipeline.

  raw text
   -> runtime budget check (Task 15)       oversized input rejected before any model call
   -> PII masking (Task 10)                the model, memory and logs only ever see masked text
   -> prompt-injection detection (Task 10)
   -> memory resolution (Task 8)           "its"/"my application" -> record id from this session
   -> routing + CrewAI crew (Task 7)       Retrieval / Lookup / Composer, output_pydantic=CrewResponse
   -> output groundedness guard (Task 10)  refuse the policy part when retrieval doesn't support it
   -> Autogen review team (Task 14)        approve unchanged or revise, structured ReviewVerdict
   -> CrewResponse validation (Task 9)
"""
from __future__ import annotations

import re
import time

import config  # noqa: F401
from agents import budget
from agents.budget import BudgetExceeded
from agents.composer_logic import render_lookup_sentences
from agents.crew import run_crew
from agents.guardrails import INJECTION_REFUSAL, detect_injection, mask_pii
from agents.schemas import CrewResponse
from rag.retrieval import IDK_ANSWER
from tools.crew_tools import last_retrieval, start_tool_log
from tools.loan_status import RECORD_ID_RE, check_loan_application_status, extract_record_id

STATUS_RE = re.compile(r"\b(status|update|progress|where is|loan amount|amount|escalat\w*|"
                       r"approved yet|disbursed yet|my application|the application|this application)\b", re.I)
REFERENCE_RE = re.compile(r"\b(it|its|it's|that|this|same|my application|the application|"
                          r"this application|that application|my loan)\b", re.I)
POLICY_HINT_RE = re.compile(r"\b(penalt|prepay|foreclos|fee|charge|emi|interest|rate|kyc|document|"
                            r"eligib|clos|fraud|dispute|unauthori[sz]ed|minimum balance|credit score|"
                            r"joint|nri|nre|nro|reapply|how long|when will|tax)\w*", re.I)
# Words that carry no extra question when they appear next to a record id.
_FILLER = set("""a an the is are was what whats what's of for me my please check can could you tell
show give find look up about on in at to it its it's this that i we our do does did and or
application loan id number record current currently status update progress latest share
hi hello thanks thank kindly regarding re account bank pan aadhaar redacted no
""".split())


def has_extra_question(query: str) -> bool:
    """True when a message with a record id ALSO asks something beyond the status itself
    (e.g. '...and can I use it to buy gold coins?'). Such messages must go to retrieval, so the
    extra question is either answered from the KB or refused by the groundedness guard —
    never silently dropped."""
    rest = STATUS_RE.sub(" ", RECORD_ID_RE.sub(" ", re.sub(r"\(regarding application [^)]*\)", " ", query)))
    words = [w for w in re.findall(r"[a-z]+", rest.lower()) if w not in _FILLER and len(w) > 2]
    return len(words) >= 2


CLARIFY_ANSWER = ("Please share the loan application ID (format CRED-LA-0000) so I can check its status.")


def record_id_from_history(history) -> str | None:
    for msg in reversed(history or []):
        ids = RECORD_ID_RE.findall(getattr(msg, "content", "") or "")
        if ids:
            return ids[-1].upper()
    return None


def _refusal(answer: str, intent: str, flags: list[str]) -> CrewResponse:
    return CrewResponse(answer=answer, intent=intent, grounded=False, guardrail_flags=flags)


def output_groundedness_guard(draft: CrewResponse, retrieval: dict | None, used_retrieval: bool,
                              flags: list[str]) -> CrewResponse:
    """Output-side guardrail: if the policy part of the answer is not supported by retrieved
    context (top-1 similarity below the calibrated threshold), the policy part is replaced by
    the refusal — whatever the composer wrote. Lookup facts (from the record) are kept."""
    if not used_retrieval:
        return draft
    if retrieval is None or not retrieval["grounded"]:
        flags.append("groundedness_refusal")
        keep = []
        if draft.record_lookup is not None:
            keep = render_lookup_sentences(check_loan_application_status(draft.record_lookup.record_id))
        # Pure policy question with no support -> the whole answer is a refusal.
        # Mixed question -> the record facts are still answered, so the intent stays as it was.
        intent = draft.intent if draft.record_lookup is not None else "refused"
        return draft.model_copy(update={"answer": " ".join(keep + [IDK_ANSWER]), "grounded": False,
                                        "sources": [], "intent": intent})
    return draft


class SupportAgent:
    def handle(self, raw_query: str, history=None, *, inject_into_draft: str | None = None,
               review: bool = True, budget_overrides: dict | None = None) -> dict:
        t0 = time.perf_counter()
        flags: list[str] = []
        out = {"raw_len": len(raw_query), "tool_calls": [], "review": None, "review_transcript": [],
               "resolved_record_id_from_memory": None, "retrieval": None, "crew_draft": None}
        with budget.request_budget(**(budget_overrides or {})) as b:
            try:
                b.check_input(raw_query)
            except BudgetExceeded as e:
                flags.append("budget_exceeded")
                return self._finish(out, _refusal(str(e), "refused", flags), "", b, t0)

            masked, pii = mask_pii(raw_query)
            flags += [f"pii_masked:{p}" for p in pii]
            out["masked_query"] = masked

            inj = detect_injection(masked)
            if inj:
                flags.append("prompt_injection_blocked")
                out["injection_matches"] = inj
                return self._finish(out, _refusal(INJECTION_REFUSAL, "refused", flags), masked, b, t0)

            query = masked
            rid = extract_record_id(query)
            if rid is None and (STATUS_RE.search(query) or REFERENCE_RE.search(query)):
                remembered = record_id_from_history(history)
                if remembered:
                    query = f"{query} (regarding application {remembered})"
                    rid = remembered
                    out["resolved_record_id_from_memory"] = remembered
            policy_hint = bool(POLICY_HINT_RE.search(query))
            if rid is None and STATUS_RE.search(query) and not policy_hint:
                flags.append("needs_record_id")
                return self._finish(out, CrewResponse(answer=CLARIFY_ANSWER, intent="clarify",
                                                      grounded=True, guardrail_flags=flags), query, b, t0)
            use_lookup = rid is not None
            use_retrieval = (not use_lookup) or policy_hint or has_extra_question(query)

            tool_log = start_tool_log()
            try:
                draft = run_crew(query, use_retrieval=use_retrieval, use_lookup=use_lookup)
            except BudgetExceeded as e:
                flags.append("budget_exceeded")
                return self._finish(out, _refusal(str(e), "refused", flags), query, b, t0)
            out["tool_calls"] = list(tool_log)
            retrieval = last_retrieval()
            out["crew_draft"] = draft.answer
            if inject_into_draft:  # test hook for Task 14: simulate an ungrounded composer claim
                draft = draft.model_copy(update={"answer": f"{draft.answer} {inject_into_draft}"})
                out["crew_draft"] = draft.answer
                flags.append("test_injected_claim")

            draft = output_groundedness_guard(draft, retrieval, use_retrieval, flags)
            out["retrieval"] = ({k: retrieval[k] for k in ("top1_similarity", "threshold", "grounded",
                                                           "sources", "contexts", "cache_hit")}
                                if retrieval else None)

            final_answer = draft.answer
            only_refusal = draft.answer.strip() == IDK_ANSWER
            if review and not only_refusal:
                from agents.review_team import review as run_review
                ctx = list(retrieval["contexts"]) if (retrieval and retrieval["grounded"]) else []
                if draft.record_lookup is not None:
                    ctx += render_lookup_sentences(
                        check_loan_application_status(draft.record_lookup.record_id))
                try:
                    verdict, transcript = run_review(draft.answer, ctx)
                except BudgetExceeded as e:
                    flags.append("budget_exceeded")
                    return self._finish(out, _refusal(str(e), "refused", flags), query, b, t0)
                out["review"] = verdict.model_dump()
                out["review_transcript"] = transcript
                final_answer = verdict.final_answer
                flags.append("review_approved" if verdict.approved else "review_revised")

            final = draft.model_copy(update={"answer": final_answer, "guardrail_flags": flags})
            return self._finish(out, final, query, b, t0)

    @staticmethod
    def _finish(out, resp: CrewResponse, model_input: str, b, t0) -> dict:
        out["response"] = CrewResponse.model_validate(resp.model_dump())  # Task 9: validate in code
        out["model_input"] = model_input
        out["budget"] = {"used_tokens": b.used_tokens, "cost_usd": b.cost_usd,
                         "max_total_tokens": b.max_total_tokens, "max_cost_usd": b.max_cost_usd}
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out


_agent: SupportAgent | None = None


def get_agent() -> SupportAgent:
    global _agent
    if _agent is None:
        _agent = SupportAgent()
    return _agent
