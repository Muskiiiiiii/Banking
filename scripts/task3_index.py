"""Task 3 on its own: chunk the KB two ways, embed with SentenceTransformers, index each
strategy in its OWN ChromaDB collection, and run one sample query against both.
Run from the project root:  python -m scripts.task3_index
"""
import config
from rag.embeddings import embed
from rag.index import build_indexes, get_collection
from rag.kb import chunk_all, load_documents

SAMPLE_QUERY = "Is there a prepayment penalty on a home loan?"


def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {config.KB_DIR}\n")

    for strategy in ("fixed", "sentence"):
        chunks = chunk_all(docs, strategy)
        print(f"--- {strategy} chunks: {len(chunks)} total; first two from {docs[0].doc_id} ---")
        for c in chunks[:2]:
            print(f"  {c.chunk_id:<36} ({len(c.text)} chars) {c.text!r}")
        print()

    build_indexes()  # embeds every chunk and collection.upsert()s into two collections

    qv = embed([SAMPLE_QUERY])[0].tolist()
    for name in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE):
        col = get_collection(name)
        res = col.query(query_embeddings=[qv], n_results=3, include=["documents", "metadatas", "distances"])
        print(f"\nSample query on {name} (count={col.count()}): {SAMPLE_QUERY!r}")
        for cid, text, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
            print(f"  cosine={1 - dist:.4f}  {cid:<36} {text[:70]}...")


if __name__ == "__main__":
    main()
