"""
Configuration settings for the Facility Support Assistant.
All configurable values are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100    # character overlap between consecutive chunks

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5      # number of chunks to retrieve per query

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash-lite"

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Building Management and Facility Support Assistant.

Answer the user's question ONLY using the provided document context.

Do not invent technical procedures, specifications, fault causes,
values, or safety instructions.

If the answer cannot be found in the provided context, respond:

"I could not find this information in the available facility documents."

Do not use outside knowledge to fill missing information.

When possible, mention the relevant source document and page."""
