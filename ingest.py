"""
ingest.py – CLI script to build (or rebuild) the FAISS knowledge base.

Usage
-----
    python ingest.py

Place PDF files in  data/documents/  before running.
"""

import sys
from pathlib import Path

# Make sure the project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.document_processor import load_documents
from src.vector_store import build_index, save_index
from src.config import DOCUMENTS_DIR, FAISS_INDEX_PATH, METADATA_PATH


def main() -> None:
    print("=" * 60)
    print("  Facility Support Assistant – Knowledge Base Builder")
    print("=" * 60)
    print(f"\nDocuments directory : {DOCUMENTS_DIR}")
    print(f"FAISS index path    : {FAISS_INDEX_PATH}")
    print(f"Metadata path       : {METADATA_PATH}\n")

    # ── Load & chunk documents ─────────────────────────────────────────────────
    try:
        chunks = load_documents()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    # ── Build FAISS index ──────────────────────────────────────────────────────
    index, metadata = build_index(chunks)

    # ── Save to disk ───────────────────────────────────────────────────────────
    save_index(index, metadata)

    unique_docs = len({m["source"] for m in metadata})
    print("\n" + "=" * 60)
    print("  Knowledge base successfully created.")
    print(f"  Documents : {unique_docs}")
    print(f"  Chunks    : {len(chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
