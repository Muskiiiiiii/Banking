"""Local SentenceTransformers embedder (free, no API key) with a call counter."""
from __future__ import annotations

import threading

import numpy as np

import config

_model = None
_lock = threading.Lock()
EMBED_STATS = {"embed_calls": 0, "texts_embedded": 0}


def get_model():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    EMBED_STATS["embed_calls"] += 1
    EMBED_STATS["texts_embedded"] += len(texts)
    return get_model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False,
                              convert_to_numpy=True)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectors are L2-normalised, so the dot product is the cosine similarity."""
    return a @ b.T
