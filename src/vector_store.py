"""
Vector store: build, save, load, and search a FAISS flat index.

Storage layout
--------------
vectorstore/
    index.faiss   – FAISS binary index (Inner Product, since embeddings are normalised)
    metadata.pkl  – list[dict] mapping integer FAISS IDs to chunk metadata
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss

from src.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K_RESULTS
from src.embeddings import embed_texts, embed_query


# ── Build & persist ─────────────────────────────────────────────────────────────

def build_index(chunks: List[Dict[str, Any]]) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Generate embeddings for all chunks and build a FAISS index.

    Parameters
    ----------
    chunks : list of dicts with keys 'text', 'source', 'page'

    Returns
    -------
    (faiss_index, metadata_list)
    """
    texts = [c["text"] for c in chunks]
    print(f"[INFO] Embedding {len(texts)} chunks …")
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    # IndexFlatIP with normalised vectors == cosine similarity search
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[INFO] FAISS index built – {index.ntotal} vectors, dim={dim}.")

    # Metadata mirrors the chunk list; FAISS integer ID == list index
    metadata = [
        {"text": c["text"], "source": c["source"], "page": c["page"]}
        for c in chunks
    ]
    return index, metadata


def save_index(index: faiss.Index, metadata: List[Dict[str, Any]]) -> None:
    """Persist the FAISS index and metadata to disk."""
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[INFO] Index saved → {FAISS_INDEX_PATH}")
    print(f"[INFO] Metadata saved → {METADATA_PATH}")


def load_index() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Load FAISS index and metadata from disk.

    Raises
    ------
    FileNotFoundError if either file is missing.
    """
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Knowledge base not found. "
            "Please build it first by clicking 'Build Knowledge Base'."
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"[INFO] Loaded index with {index.ntotal} vectors.")
    return index, metadata


# ── Search ──────────────────────────────────────────────────────────────────────

def search(query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
    """
    Embed *query* and return the top-k most similar chunks.

    Each result dict contains: 'text', 'source', 'page', 'score'.
    """
    index, metadata = load_index()

    query_vec = embed_query(query)          # shape (1, dim)
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:                       # FAISS sentinel for "not found"
            continue
        chunk = dict(metadata[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    return results


# ── Utility ─────────────────────────────────────────────────────────────────────

def index_stats() -> Dict[str, int]:
    """
    Return basic stats about the persisted index without loading embeddings.
    Returns {'vectors': 0, 'documents': 0} if no index exists.
    """
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        return {"vectors": 0, "documents": 0}

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    unique_docs = len({m["source"] for m in metadata})
    return {"vectors": index.ntotal, "documents": unique_docs}
