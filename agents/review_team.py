"""Task 14 — Autogen review stage after the CrewAI Composer's draft.

Team: RoundRobinGroupChat([Policy-Compliance-Reviewer, Final-Editor], max_turns=2).
  * Policy-Compliance-Reviewer checks every sentence of the draft against the retrieved
    context (claim-level support: embedding similarity + numbers must appear in context)
    and lists the unsupported ones.
  * Final-Editor reads the reviewer's findings and emits a structured ReviewVerdict
    (output_content_type=ReviewVerdict). The team registers
    StructuredMessage[ReviewVerdict] via custom_message_types, otherwise Autogen raises
    "ValueError: Message type ... is not registered".

Under MOCK_LLM both agents run on MockReviewClient, a deterministic implementation of
autogen_core's ChatCompletionClient interface (no key, no network).
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncGenerator, Mapping, Optional, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import (ChatCompletionClient, CreateResult, LLMMessage, ModelInfo,
                                 RequestUsage)
from pydantic import BaseModel

import config
from agents import budget
from agents.schemas import ReviewVerdict
from rag.grounding import strip_citation, support_report
from rag.kb import split_sentences
from rag.retrieval import IDK_ANSWER

DRAFT_RE = re.compile(r"DRAFT:\s*<<<(.*?)>>>", re.DOTALL)
CONTEXT_RE = re.compile(r"CONTEXT:\s*<<<(.*?)>>>", re.DOTALL)
UNSUPPORTED_RE = re.compile(r"^UNSUPPORTED: (.*?) \|\|", re.MULTILINE)
REVIEW_STATS = {"reviewer_calls": 0, "editor_calls": 0}

REVIEWER_SYSTEM = ("You are Cred's Policy-Compliance-Reviewer. Check each sentence of the DRAFT against "
                   "the CONTEXT. List every sentence that the CONTEXT does not support, one per line as "
                   "'UNSUPPORTED: <sentence> || <why>'. If all are supported, say 'ALL CLAIMS SUPPORTED'.")
EDITOR_SYSTEM = ("You are Cred's Final-Editor. Using the reviewer's findings, approve the DRAFT unchanged "
                 "if every claim is supported; otherwise remove the unsupported sentences. Reply only with "
                 "JSON matching the ReviewVerdict schema (approved, final_answer, reason).")


def _text(m: LLMMessage) -> str:
    c = getattr(m, "content", "")
    return c if isinstance(c, str) else json.dumps(c, default=str)


class MockReviewClient(ChatCompletionClient):
    """Deterministic ChatCompletionClient for the two review agents."""

    def __init__(self, role: str):
        self.role = role
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = [],
                     tool_choice: Any = "auto", json_output: Optional[bool | type[BaseModel]] = None,
                     extra_create_args: Mapping[str, Any] = {},
                     cancellation_token: Optional[CancellationToken] = None) -> CreateResult:
        convo = "\n".join(_text(m) for m in messages)
        task = next((_text(m) for m in messages if "DRAFT:" in _text(m)), "")
        draft = (DRAFT_RE.search(task).group(1).strip() if DRAFT_RE.search(task) else "")
        context = (CONTEXT_RE.search(task).group(1).strip() if CONTEXT_RE.search(task) else "")
        ctx_items = [c for c in context.split("\n") if c.strip()]

        if self.role == "reviewer":
            REVIEW_STATS["reviewer_calls"] += 1
            rep = support_report(draft, ctx_items)
            bad = [r for r in rep if not r["supported"]]
            if not bad:
                out = f"ALL CLAIMS SUPPORTED ({len(rep)} of {len(rep)} sentences checked against the context)."
            else:
                lines = [f"UNSUPPORTED: {r['sentence']} || best_similarity={r['best_sim']}"
                         f"{', numbers not in context=' + str(r['missing_numbers']) if r['missing_numbers'] else ''}"
                         for r in bad]
                out = (f"{len(bad)} of {len(rep)} sentences are not supported by the context:\n"
                       + "\n".join(lines))
        else:
            REVIEW_STATS["editor_calls"] += 1
            # Read the reviewer's message (the last message that is not the task itself).
            findings = next((_text(m) for m in reversed(messages)
                             if "UNSUPPORTED:" in _text(m) or "ALL CLAIMS SUPPORTED" in _text(m)), "")
            flagged = UNSUPPORTED_RE.findall(findings)
            if not flagged:
                verdict = ReviewVerdict(approved=True, final_answer=draft,
                                        reason="Reviewer found every claim supported by the retrieved "
                                               "context; draft approved unchanged.")
            else:
                kept = []
                for s in split_sentences(draft):
                    core = strip_citation(s)
                    if core not in flagged:
                        kept.append(s)
                final = " ".join(kept) if kept else IDK_ANSWER
                verdict = ReviewVerdict(approved=False, final_answer=final,
                                        reason=f"Removed {len(flagged)} unsupported claim(s): "
                                               + " | ".join(flagged))
            out = verdict.model_dump_json()

        budget.charge(f"review:{self.role}", convo, out)
        p, c = budget.estimate_tokens(convo), budget.estimate_tokens(out)
        self._usage = RequestUsage(prompt_tokens=self._usage.prompt_tokens + p,
                                   completion_tokens=self._usage.completion_tokens + c)
        return CreateResult(finish_reason="stop", content=out,
                            usage=RequestUsage(prompt_tokens=p, completion_tokens=c), cached=False)

    async def create_stream(self, messages, **kwargs) -> AsyncGenerator[Any, None]:
        yield await self.create(messages, **kwargs)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(self, messages, *, tools=[]) -> int:
        return budget.estimate_tokens("\n".join(_text(m) for m in messages))

    def remaining_tokens(self, messages, *, tools=[]) -> int:
        return 128_000 - self.count_tokens(messages)

    @property
    def capabilities(self):  # deprecated alias kept for interface completeness
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True,
                         family="unknown", structured_output=True)


def _client(role: str) -> ChatCompletionClient:
    if config.MOCK_LLM:
        return MockReviewClient(role)
    from autogen_ext.models.openai import OpenAIChatCompletionClient  # pragma: no cover
    return OpenAIChatCompletionClient(model=config.REAL_LLM_MODEL)


def build_review_team() -> RoundRobinGroupChat:
    reviewer = AssistantAgent("Policy_Compliance_Reviewer", model_client=_client("reviewer"),
                              system_message=REVIEWER_SYSTEM)
    editor = AssistantAgent("Final_Editor", model_client=_client("editor"),
                            system_message=EDITOR_SYSTEM, output_content_type=ReviewVerdict)
    return RoundRobinGroupChat([reviewer, editor], max_turns=2,
                               custom_message_types=[StructuredMessage[ReviewVerdict]])


async def areview(draft: str, context_items: list[str]) -> tuple[ReviewVerdict, list[str]]:
    team = build_review_team()
    ctx = "\n".join(" ".join(c.split()) for c in context_items)
    task = f"Review this draft answer.\nDRAFT: <<<{draft}>>>\nCONTEXT: <<<{ctx}>>>"
    result = await team.run(task=task)
    transcript = [f"{m.source}: {getattr(m, 'content', '')}" for m in result.messages]
    last = result.messages[-1]
    if not isinstance(last, StructuredMessage):
        raise RuntimeError(f"Final-Editor did not return a structured verdict: {type(last).__name__}")
    verdict = ReviewVerdict.model_validate(last.content.model_dump())
    return verdict, transcript


def review(draft: str, context_items: list[str]) -> tuple[ReviewVerdict, list[str]]:
    """Sync wrapper (the pipeline runs in a worker thread with no running event loop)."""
    return asyncio.run(areview(draft, context_items))
