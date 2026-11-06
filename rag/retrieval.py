"""Task 4 + Task 16 — retrieval, grounded generation with an "I don't know" fallback,
and an in-memory response cache keyed by normalised query text."""
from __future__ import annotations

import re
import threading
import time

import config
from rag import llm
from rag.embeddings import cosine_matrix, embed
from rag.index import get_collection
from rag.kb import expand_to_sentences

IDK_ANSWER = ("I don't know — Cred's policy documents do not cover this question. "
              "Please contact a support specialist.")

GROUNDED_PROMPT = """You are Cred's loan-support assistant.
Answer the QUESTION using ONLY the CONTEXT below. Quote figures exactly as written.
If the CONTEXT does not contain the answer, reply exactly: "{idk}"
Cite the source document id in square brackets after each sentence.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


# The brand name appears across the whole KB ("Cred app", "Cred-partner card"), so it carries
# no topic signal but pulls ANY question that mentions it toward the KB. Measured effect:
# "How do I invest in Bitcoin through Cred?" scored 0.430 (above the 0.42 threshold) with the
# word and 0.211 without it. It is therefore removed from the text that gets embedded.
BRAND_RE = re.compile(r"\bcred(?:'s)?\b(?!-)", re.IGNORECASE)


def retrieval_text(query: str) -> str:
    return " ".join(BRAND_RE.sub(" ", query).split())


def retrieve(query: str, collection_name: str = config.DEPLOYED_COLLECTION,
             k: int = config.TOP_K) -> list[dict]:
    col = get_collection(collection_name)
    qv = embed([retrieval_text(query)])[0].tolist()
    res = col.query(query_embeddings=[qv], n_results=k,
                    include=["documents", "metadatas", "distances"])
    hits = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                    res["distances"][0]):
        hits.append({"chunk_id": cid, "doc_id": meta["doc_id"], "title": meta.get("title", ""),
                     "text": doc, "similarity": round(1.0 - float(dist), 4),
                     "sentences": expand_to_sentences(doc, meta)})
    return hits


def _mock_extractive_answer(query: str, hits: list[dict]) -> str:
    """Deterministic stand-in for the LLM: pick the context sentences closest to the
    question (top sentence + any within 0.08 of it, max 3) and cite their documents.
    It can only ever output sentences that exist in the retrieved context."""
    cands = []
    seen_c = set()
    for h in hits:
        for s in h["sentences"]:
            if s not in seen_c:
                seen_c.add(s)
                cands.append((s, h["doc_id"]))
    if not cands:
        return IDK_ANSWER
    sims = cosine_matrix(embed([query]), embed([c[0] for c in cands]))[0]
    order = sorted(range(len(cands)), key=lambda i: -sims[i])
    best = sims[order[0]]
    chosen, seen = [], set()
    for i in order:
        if sims[i] < best - 0.08 or len(chosen) == 3:
            break
        if cands[i][0] not in seen:
            seen.add(cands[i][0])
            chosen.append(i)
    chosen.sort()  # keep reading order stable
    return " ".join(_cite(cands[i][0], cands[i][1]) for i in chosen)


def _cite(sentence: str, doc_id: str) -> str:
    """Put the citation before the final full stop: '... guidelines [08_prepayment_penalty].'"""
    if sentence and sentence[-1] in ".!?":
        return f"{sentence[:-1]} [{doc_id}]{sentence[-1]}"
    return f"{sentence} [{doc_id}]"


def normalize_query(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s%-]", " ", q)
    return " ".join(q.split())


class GroundedGenerator:
    """Grounded generation with an in-memory cache keyed by normalised query text."""

    def __init__(self, collection_name: str = config.DEPLOYED_COLLECTION, k: int = config.TOP_K,
                 threshold: float = config.IDK_THRESHOLD, use_cache: bool = True):
        self.collection_name, self.k, self.threshold = collection_name, k, threshold
        self.use_cache = use_cache
        self._cache: dict[tuple, dict] = {}
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0}

    def clear_cache(self):
        with self._lock:
            self._cache.clear()

    def answer(self, query: str) -> dict:
        key = (normalize_query(query), self.collection_name, self.k)
        t0 = time.perf_counter()
        if self.use_cache:
            with self._lock:
                if key in self._cache:
                    self.stats["hits"] += 1
                    out = dict(self._cache[key], cache_hit=True)
                    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                    return out
        self.stats["misses"] += 1
        hits = retrieve(query, self.collection_name, self.k)
        top1 = hits[0]["similarity"] if hits else 0.0
        grounded = top1 >= self.threshold
        if grounded:
            context = "\n".join(f"[{h['doc_id']}] {' '.join(h['sentences'])}" for h in hits)
            prompt = GROUNDED_PROMPT.format(idk=IDK_ANSWER, context=context, question=query)
            text = llm.complete(prompt, purpose="grounded_generation",
                                mock_fn=lambda: _mock_extractive_answer(query, hits))
        else:
            text = IDK_ANSWER  # fallback: no LLM call when retrieval does not support the question
        out = {"query": query, "answer": text, "grounded": grounded, "top1_similarity": top1,
               "threshold": self.threshold, "sources": sorted({h["doc_id"] for h in hits}) if grounded else [],
               "contexts": list(dict.fromkeys(s for h in hits for s in h["sentences"])), "hits": hits, "cache_hit": False}
        if self.use_cache:
            with self._lock:
                self._cache[key] = out
        out = dict(out, latency_ms=round((time.perf_counter() - t0) * 1000, 3))
        return out


_default: GroundedGenerator | None = None


def default_generator() -> GroundedGenerator:
    global _default
    if _default is None:
        _default = GroundedGenerator()
    return _default
