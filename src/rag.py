"""
RAG pipeline: retrieves relevant chunks via FAISS and generates a
grounded answer using the Gemini API.
"""

from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT, TOP_K_RESULTS
from src.vector_store import search


# ── Gemini setup ────────────────────────────────────────────────────────────────

def _get_gemini_model():
    """Configure and return the Gemini generative model."""
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file and restart the application."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )


# ── Context builder ─────────────────────────────────────────────────────────────

def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a clearly delimited context block."""
    if not chunks:
        return "No relevant document sections were found."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i} | Source: {chunk['source']} | Page: {chunk['page']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _build_history_text(chat_history: Optional[List[Dict[str, str]]]) -> str:
    """Render recent conversation history as plain text for inclusion in the prompt."""
    if not chat_history:
        return ""
    lines = ["Recent conversation:"]
    for turn in chat_history[-6:]:   # last 3 exchanges (6 messages)
        role = turn.get("role", "user").capitalize()
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


# ── Sources deduplication ───────────────────────────────────────────────────────

def _deduplicate_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return unique (source, page) pairs preserving retrieval order."""
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": chunk["source"], "page": str(chunk["page"])})
    return sources


# ── Main RAG function ───────────────────────────────────────────────────────────

def answer_question(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Full RAG flow:
        1. Embed the question with MiniLM
        2. FAISS similarity search → top-k chunks
        3. Build a grounded prompt with context + history
        4. Call Gemini
        5. Return (answer_text, sources_list)

    Parameters
    ----------
    question     : the user's question string
    chat_history : list of {'role': 'user'|'assistant', 'content': '...'} dicts

    Returns
    -------
    (answer, sources)
        answer  – string returned by Gemini
        sources – list of {'source': filename, 'page': page_number} dicts
    """
    question = question.strip()
    if not question:
        return "Please enter a question.", []

    # 1 & 2: retrieve relevant chunks
    chunks = search(question, k=TOP_K_RESULTS)

    # 3: build the full prompt
    context_block = _build_context(chunks)
    history_block = _build_history_text(chat_history)

    prompt_parts = []
    if history_block:
        prompt_parts.append(history_block)
    prompt_parts.append(f"Relevant document excerpts:\n\n{context_block}")
    prompt_parts.append(f"User question: {question}")

    full_prompt = "\n\n".join(prompt_parts)

    # 4: call Gemini
    model = _get_gemini_model()
    try:
        response = model.generate_content(full_prompt)
        answer = response.text.strip()
    except Exception as exc:
        answer = f"Gemini API error: {exc}"
        return answer, []

    # 5: sources
    sources = _deduplicate_sources(chunks)
    return answer, sources
