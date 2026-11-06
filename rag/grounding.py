"""Claim-level groundedness: is each answer sentence supported by the retrieved context?

A sentence is SUPPORTED when (a) its best cosine similarity to any context sentence is
>= CLAIM_SUPPORT_THRESHOLD and (b) every number it states (amounts, %, days) also
appears somewhere in the context. Check (b) is what catches a fabricated figure that is
otherwise worded like a real policy sentence (high embedding similarity, wrong number).

Used by the output-side guardrail (Task 10), the Autogen Policy-Compliance-Reviewer
(Task 14) and the Grounding score of the evaluation (Task 13).
"""
from __future__ import annotations

import re

import config
from rag.embeddings import cosine_matrix, embed
from rag.kb import split_sentences
from rag.retrieval import IDK_ANSWER

# System refusal text is not a factual claim, so it is never flagged as unsupported.
NON_CLAIM_SENTENCES = set(split_sentences(IDK_ANSWER))

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CITATION = re.compile(r"\s*\[[^\]]+\](?=[.!?]?\s*$)")


def numbers_in(text: str) -> set[str]:
    out = set()
    for m in _NUM.findall(text):
        v = m.replace(",", "").rstrip(".")
        if v:
            out.add(v)
    return out


def strip_citation(sentence: str) -> str:
    return _CITATION.sub("", sentence).strip()


def support_report(answer: str, context_texts: list[str],
                   threshold: float = config.CLAIM_SUPPORT_THRESHOLD) -> list[dict]:
    claims = [strip_citation(s) for s in split_sentences(answer)]
    claims = [c for c in claims if c and c not in NON_CLAIM_SENTENCES]
    ctx_sents = [s for t in context_texts for s in split_sentences(t)]
    if not claims:
        return []
    if not ctx_sents:
        return [{"sentence": c, "best_sim": 0.0, "missing_numbers": sorted(numbers_in(c)),
                 "supported": False} for c in claims]
    sims = cosine_matrix(embed(claims), embed(ctx_sents))
    ctx_numbers = numbers_in(" ".join(context_texts))
    report = []
    for i, c in enumerate(claims):
        best = float(sims[i].max())
        missing = sorted(numbers_in(c) - ctx_numbers)
        report.append({"sentence": c, "best_sim": round(best, 4), "missing_numbers": missing,
                       "supported": best >= threshold and not missing})
    return report


def grounding_score(answer: str, context_texts: list[str]) -> float:
    rep = support_report(answer, context_texts)
    if not rep:
        return 1.0
    return round(sum(r["supported"] for r in rep) / len(rep), 4)
