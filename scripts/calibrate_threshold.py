"""Task 4 — empirically calibrate the "I don't know" threshold.

Measures top-1 cosine similarity for in-scope and deliberately out-of-scope queries on
BOTH collections, then reports the gap between the two clusters. The chosen value is
stored in config.IDK_THRESHOLD.
Run: python -m scripts.calibrate_threshold
"""
import config  # noqa: F401  (sets telemetry env vars first)
from rag.embeddings import embed
from rag.index import build_indexes, get_collection
from rag.retrieval import retrieval_text, retrieve

IN_SCOPE = [
    "What is the minimum income to get a personal loan?",
    "Is there a penalty for prepaying my floating-rate home loan?",
    "What documents do I need for KYC?",
    "How long does a fraud dispute take to resolve?",
    "What is the annual fee on the credit card?",
    "Can an NRI open an NRE account?",
]
OUT_OF_SCOPE = [
    "What is the best pizza place in Bengaluru?",
    "Who won the cricket world cup final?",
    "How do I bake sourdough bread at home?",
    "Can you recommend a good laptop for gaming?",
    "What is the weather like in Goa this weekend?",
    # near-domain (finance-flavoured) out-of-scope queries — the hard cases
    "What is the current price of gold?",
    "How do I file my income tax return?",
    "Tell me about your travel insurance plans",
    # brand-word trap: scored 0.430 before the brand name was stripped from retrieval text
    "How do I invest in Bitcoin through Cred?",
]

if __name__ == "__main__":
    build_indexes()
    for cname in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE):
        print(f"\n=== {cname} ===")
        ins, outs = [], []
        for q in IN_SCOPE:
            s = retrieve(q, cname, 1)[0]["similarity"]; ins.append(s)
            print(f"  IN   top1={s:.4f}  {q}")
        for q in OUT_OF_SCOPE:
            s = retrieve(q, cname, 1)[0]["similarity"]; outs.append(s)
            print(f"  OUT  top1={s:.4f}  {q}")
        lo_in, hi_out = min(ins), max(outs)
        print(f"  in-scope range  [{min(ins):.4f}, {max(ins):.4f}]")
        print(f"  out-of-scope range [{min(outs):.4f}, {max(outs):.4f}]")
        print(f"  gap = ({hi_out:.4f}, {lo_in:.4f})  midpoint = {(hi_out + lo_in) / 2:.4f}")
    print(f"\nConfigured IDK_THRESHOLD = {config.IDK_THRESHOLD}")

    # Brand-word check: the same query embedded WITH and WITHOUT the word "Cred".
    q = "How do I invest in Bitcoin through Cred?"
    print(f"\nBrand-word check for {q!r} (retrieval text without brand: {retrieval_text(q)!r})")
    for cname in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE):
        col = get_collection(cname)
        raw = 1 - col.query(query_embeddings=[embed([q])[0].tolist()], n_results=1)["distances"][0][0]
        stripped = retrieve(q, cname, 1)[0]["similarity"]
        print(f"  {cname:<18} with 'Cred' = {raw:.4f}   brand removed = {stripped:.4f}   "
              f"threshold = {config.IDK_THRESHOLD}")
