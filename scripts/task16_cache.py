"""Task 16 on its own: in-memory response cache for grounded generation, keyed by
normalised query text, with before/after evidence (call counters + timing).
Run: python -m scripts.task16_cache
"""
import time

import config  # noqa: F401
from agents.memory import ConversationManager
from rag import llm
from rag.embeddings import EMBED_STATS
from rag.index import ensure_indexes
from rag.retrieval import GroundedGenerator, normalize_query


def main():
    ensure_indexes()
    gen = GroundedGenerator()
    q1 = "What is the prepayment penalty on a personal loan?"
    q2 = "  what is the PREPAYMENT penalty on a personal loan  "
    q3 = "What is the late-payment fee on the credit card?"
    print("Cache key = normalise(query) + collection + top-k")
    print(f"  normalise({q1!r}) = {normalize_query(q1)!r}")
    print(f"  normalise({q2!r}) = {normalize_query(q2)!r}\n")

    print(f"{'call':<40}{'cache_hit':<11}{'LLM calls':<11}{'embed calls':<13}{'time (ms)':>10}")
    for label, q in (("1. first ask (cold)", q1), ("2. identical repeat", q1),
                     ("3. case/space/punctuation variant", q2), ("4. a different question", q3)):
        l0, e0 = llm.LLM_STATS["calls"], EMBED_STATS["embed_calls"]
        t0 = time.perf_counter()
        r = gen.answer(q)
        dt = (time.perf_counter() - t0) * 1000
        print(f"{label:<40}{str(r['cache_hit']):<11}{'+' + str(llm.LLM_STATS['calls'] - l0):<11}"
              f"{'+' + str(EMBED_STATS['embed_calls'] - e0):<13}{dt:>10.3f}")
    print(f"\ncache stats: {gen.stats}")

    print("\nThrough the full crew (RAG tool reports whether the cache was hit):")
    cm = ConversationManager()
    for i in range(2):
        l0 = llm.LLM_STATS["by_purpose"].get("grounded_generation", 0)
        r = cm.ask(f"cache-{i}", "What is the joining fee on the credit card?")
        print(f"  run {i + 1}: cache_hit={r['retrieval']['cache_hit']}  grounded-generation LLM calls "
              f"+{llm.LLM_STATS['by_purpose'].get('grounded_generation', 0) - l0}  "
              f"answer: {r['response'].answer[:70]}...")


if __name__ == "__main__":
    main()
