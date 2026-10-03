"""
Embeddings: loads the SentenceTransformer model once and exposes
helper functions for encoding documents and queries.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

# ── Singleton model instance ────────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the (cached) SentenceTransformer model."""
    global _model
    if _model is None:
        print(f"[INFO] Loading embedding model '{EMBEDDING_MODEL}' …")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("[INFO] Embedding model loaded.")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised embedding vectors.

    Returns
    -------
    np.ndarray of shape (N, dim) with dtype float32.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,   # cosine similarity via inner product
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string.

    Returns
    -------
    np.ndarray of shape (1, dim) with dtype float32 – ready for FAISS search.
    """
    return embed_texts([query])
