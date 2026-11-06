"""Task 5 on its own: document-level precision / recall for BOTH collections on the same
queries used in Task 4. Chunks are mapped back to their parent document and de-duplicated
before scoring.
Run: python -m scripts.task5_eval_chunking
"""
import config
from rag.index import ensure_indexes
from rag.retrieval import retrieve
from scripts.task4_grounded import EVAL_QUERIES


def score_collection(cname: str) -> tuple[float, float]:
    print(f"\n{'=' * 90}\n{cname}   (top-k = {config.TOP_K} chunks)\n{'=' * 90}")
    ps, rs = [], []
    for i, (q, relevant) in enumerate(EVAL_QUERIES.items(), 1):
        hits = retrieve(q, cname, config.TOP_K)
        chunk_docs = [h["doc_id"] for h in hits]
        retrieved = list(dict.fromkeys(chunk_docs))  # dedup, keep rank order
        tp = [d for d in retrieved if d in relevant]
        p, r = len(tp) / len(retrieved), len(tp) / len(relevant)
        ps.append(p); rs.append(r)
        print(f"Q{i}: {q}\n    chunks -> parent docs : {chunk_docs}\n"
              f"    deduped retrieved     : {retrieved}\n    relevant              : {sorted(relevant)}\n"
              f"    precision = {len(tp)}/{len(retrieved)} = {p:.3f}    recall = {len(tp)}/{len(relevant)} = {r:.3f}")
    mp, mr = sum(ps) / len(ps), sum(rs) / len(rs)
    print(f"\n  MEAN precision = ({' + '.join(f'{x:.3f}' for x in ps)}) / {len(ps)} = {mp:.3f}")
    print(f"  MEAN recall    = ({' + '.join(f'{x:.3f}' for x in rs)}) / {len(rs)} = {mr:.3f}")
    return mp, mr


def main():
    ensure_indexes()
    results = {c: score_collection(c) for c in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE)}
    print(f"\n{'=' * 90}\nSUMMARY")
    for c, (p, r) in results.items():
        print(f"  {c:<18} mean precision = {p:.3f}   mean recall = {r:.3f}")
    print(f"  deployed collection (config.DEPLOYED_COLLECTION) = {config.DEPLOYED_COLLECTION}")


if __name__ == "__main__":
    main()
