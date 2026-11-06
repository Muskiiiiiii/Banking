"""Task 3 — embed both chunk sets and index each in its OWN ChromaDB collection."""
from __future__ import annotations

import chromadb
from chromadb.config import Settings

import config
from rag.embeddings import embed
from rag.kb import Document, chunk_all, chunk_fixed, chunk_sentences, load_documents

_client = None

STRATEGY_TO_COLLECTION = {"fixed": config.COLLECTION_FIXED, "sentence": config.COLLECTION_SENTENCE}


def get_client():
    global _client
    if _client is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR),
                                            settings=Settings(anonymized_telemetry=False))
    return _client


def get_collection(name: str):
    # cosine space: Chroma returns distance = 1 - cosine_similarity
    return get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(collection_name: str, chunks) -> int:
    if not chunks:
        return 0
    col = get_collection(collection_name)
    vecs = embed([c.text for c in chunks])
    col.upsert(ids=[c.chunk_id for c in chunks], documents=[c.text for c in chunks],
               embeddings=vecs.tolist(), metadatas=[c.metadata for c in chunks])
    return len(chunks)


def build_indexes(verbose: bool = True) -> dict:
    """(Re)build both collections from kb/ only. Each collection is deleted and recreated
    first, so the index always matches the knowledge base exactly — documents added at
    runtime through POST /add-document do not survive a rebuild."""
    docs = load_documents()
    counts = {}
    client = get_client()
    for strategy, cname in STRATEGY_TO_COLLECTION.items():
        if cname in [c.name if hasattr(c, "name") else c for c in client.list_collections()]:
            client.delete_collection(cname)
        chunks = chunk_all(docs, strategy)
        counts[cname] = upsert_chunks(cname, chunks)
        if verbose:
            print(f"[index] {cname:<18} strategy={strategy:<8} chunks={counts[cname]} "
                  f"(from {len(docs)} documents)")
    return counts


def add_document(doc_id: str, title: str, text: str) -> dict:
    """Used by POST /add-document: chunk the new doc both ways and upsert into both collections."""
    doc = Document(doc_id=doc_id, title=title, text=" ".join(text.split()))
    return {
        config.COLLECTION_FIXED: upsert_chunks(config.COLLECTION_FIXED, chunk_fixed(doc)),
        config.COLLECTION_SENTENCE: upsert_chunks(config.COLLECTION_SENTENCE, chunk_sentences(doc)),
    }


def ensure_indexes() -> None:
    for cname in STRATEGY_TO_COLLECTION.values():
        if get_collection(cname).count() == 0:
            build_indexes(verbose=False)
            return
