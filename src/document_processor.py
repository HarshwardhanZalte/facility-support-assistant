"""
Document processor: reads PDFs from the documents directory,
extracts text page-by-page with PyMuPDF, and splits into overlapping chunks.
"""

from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF

from src.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def extract_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Open a PDF and return a list of {text, source, page} dicts – one per page.
    Returns an empty list if the PDF cannot be read or contains no text.
    """
    pages = []
    filename = pdf_path.name

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"[WARN] Could not open {filename}: {exc}")
        return pages

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "text": text,
                    "source": filename,
                    "page": page_num + 1,   # 1-indexed, human-friendly
                })
        except Exception as exc:
            print(f"[WARN] Could not read page {page_num + 1} of {filename}: {exc}")

    doc.close()
    return pages


def split_into_chunks(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split a single page's text into overlapping character-level chunks.
    Each chunk retains the source filename and page number.
    """
    text = page["text"]
    source = page["source"]
    page_num = page["page"]

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page_num,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def load_documents() -> List[Dict[str, Any]]:
    """
    Walk through DOCUMENTS_DIR, process every PDF, and return all chunks.
    """
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{DOCUMENTS_DIR}'. "
            "Upload documents before building the knowledge base."
        )

    all_chunks: List[Dict[str, Any]] = []
    processed = 0

    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        if not pages:
            print(f"[WARN] '{pdf_path.name}' yielded no extractable text – skipped.")
            continue

        for page in pages:
            all_chunks.extend(split_into_chunks(page))

        processed += 1
        print(f"[INFO] Processed '{pdf_path.name}' – {len(pages)} pages.")

    if not all_chunks:
        raise ValueError(
            "PDFs were found but no text could be extracted. "
            "Ensure the documents contain selectable text (not scanned images)."
        )

    print(f"[INFO] Total chunks generated: {len(all_chunks)} from {processed} PDF(s).")
    return all_chunks
