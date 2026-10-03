"""
app.py – Streamlit front-end for the Facility Support Assistant.

Run with:
    streamlit run app.py
"""

import sys
import shutil
from pathlib import Path

import streamlit as st

# ── Ensure project root is importable ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DOCUMENTS_DIR, GEMINI_API_KEY
from src.document_processor import load_documents
from src.vector_store import build_index, save_index, index_stats
from src.rag import answer_question

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Facility Support Assistant",
    page_icon="🏢",
    layout="wide",
)

# ── Session state defaults ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, sources}
if "kb_built" not in st.session_state:
    st.session_state.kb_built = False


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar – Document Management
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📁 Document Management")
    st.markdown("---")

    # ── API key warning ────────────────────────────────────────────────────────
    if not GEMINI_API_KEY:
        st.error(
            "⚠️ **GEMINI_API_KEY** is missing.\n\n"
            "Create a `.env` file in the project root with:\n"
            "```\nGEMINI_API_KEY=your_api_key_here\n```"
        )

    # ── PDF upload ─────────────────────────────────────────────────────────────
    st.subheader("Upload PDF")
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for uf in uploaded_files:
            dest = DOCUMENTS_DIR / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved.append(uf.name)
        st.success(f"Saved: {', '.join(saved)}")

    st.markdown("---")

    # ── Build knowledge base ───────────────────────────────────────────────────
    if st.button("🔨 Build Knowledge Base", use_container_width=True):
        pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
        if not pdf_files:
            st.error("No PDFs found in `data/documents/`. Please upload files first.")
        else:
            with st.spinner("Processing documents and building index …"):
                try:
                    chunks = load_documents()
                    index, metadata = build_index(chunks)
                    save_index(index, metadata)
                    unique_docs = len({m["source"] for m in metadata})
                    st.session_state.kb_built = True
                    st.success(
                        f"✅ Knowledge base successfully created.\n\n"
                        f"**Documents:** {unique_docs}  \n"
                        f"**Chunks:** {len(chunks)}"
                    )
                except FileNotFoundError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    st.markdown("---")

    # ── Index stats ────────────────────────────────────────────────────────────
    st.subheader("Index Statistics")
    stats = index_stats()
    col1, col2 = st.columns(2)
    col1.metric("Documents indexed", stats["documents"])
    col2.metric("Chunks indexed", stats["vectors"])

    # ── List indexed documents ─────────────────────────────────────────────────
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf")) if DOCUMENTS_DIR.exists() else []
    if pdf_files:
        st.markdown("**Files in documents folder:**")
        for p in pdf_files:
            st.markdown(f"- 📄 `{p.name}`")


# ══════════════════════════════════════════════════════════════════════════════
#  Main area – Chat
# ══════════════════════════════════════════════════════════════════════════════

st.title("🏢 Facility Support Assistant")
st.caption(
    "Ask questions about equipment problems, procedures, or SOPs. "
    "Answers are grounded exclusively in your uploaded facility documents."
)

# ── Clear chat button ──────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Render sources beneath assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("**Sources**")
            for src in msg["sources"]:
                st.markdown(f"📄 `{src['source']}` — Page {src['page']}")

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a facility question, e.g. 'Chiller tripping on high pressure…'")

if user_input:
    # ── Validate state ─────────────────────────────────────────────────────────
    if not GEMINI_API_KEY:
        st.error("Cannot answer: GEMINI_API_KEY is not configured. See the sidebar.")
        st.stop()

    if not user_input.strip():
        st.warning("Please enter a question.")
        st.stop()

    # ── Append user message ────────────────────────────────────────────────────
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Generate answer ────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Searching documents …"):
            # Build chat history list for the RAG pipeline
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]   # exclude the current user msg
            ]

            try:
                answer, sources = answer_question(user_input, chat_history=history)
            except FileNotFoundError as exc:
                answer = f"⚠️ {exc}"
                sources = []
            except EnvironmentError as exc:
                answer = f"⚠️ {exc}"
                sources = []
            except Exception as exc:
                answer = f"⚠️ An unexpected error occurred: {exc}"
                sources = []

        # Display answer
        st.markdown(answer)

        # Display sources
        if sources:
            st.markdown("**Sources**")
            for src in sources:
                st.markdown(f"📄 `{src['source']}` — Page {src['page']}")

    # ── Persist assistant message ──────────────────────────────────────────────
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
