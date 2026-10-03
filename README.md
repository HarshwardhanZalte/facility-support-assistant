# 🏢 Building Management / Facility Support Assistant

A local RAG (Retrieval-Augmented Generation) chatbot that lets facility operators upload PDF manuals and SOPs, then ask natural-language questions about equipment problems. Answers are grounded **exclusively** in the uploaded documents.

---

## Features

- 📄 Upload PDF manuals and SOPs through the web UI
- 🔍 Semantic search over document content using FAISS
- 🤖 Grounded answers from Gemini — no hallucinated procedures
- 📌 Every answer cites the exact source document and page number
- 💬 Multi-turn conversation with context-aware follow-up questions
- 🚫 Explicit "not found" response when the answer is absent from the documents
- 🖥️ Runs entirely locally (no cloud infrastructure needed)

---

## Architecture

```
User question
     ↓
MiniLM embedding  (all-MiniLM-L6-v2)
     ↓
FAISS similarity search  → top-5 document chunks
     ↓
Context block  (chunk text + source + page)
     ↓
Gemini API  (strict system prompt)
     ↓
Grounded answer + source citations
```

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| PDF extraction | PyMuPDF (fitz) |
| Embeddings | Sentence Transformers – `all-MiniLM-L6-v2` |
| Vector search | FAISS (faiss-cpu) |
| LLM | Google Gemini (`gemini-1.5-flash`) |
| Config | python-dotenv |
| Language | Python 3.10+ |

---

## Folder Structure

```
facility-support-assistant/
│
├── data/
│   └── documents/          ← place your PDF files here
│
├── vectorstore/
│   ├── index.faiss         ← FAISS index (auto-generated)
│   └── metadata.pkl        ← chunk metadata (auto-generated)
│
├── src/
│   ├── __init__.py
│   ├── config.py           ← all configurable settings
│   ├── document_processor.py  ← PDF extraction & chunking
│   ├── embeddings.py       ← SentenceTransformer wrapper
│   ├── vector_store.py     ← FAISS build / load / search
│   └── rag.py              ← RAG pipeline & Gemini call
│
├── app.py                  ← Streamlit application
├── ingest.py               ← CLI knowledge-base builder
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone / download the project

```bash
git clone <repo-url>
cd facility-support-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run will also download the `all-MiniLM-L6-v2` model (~90 MB) automatically.

---

## Gemini API Key Setup

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and create an API key.
2. Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

3. Open `.env` and replace the placeholder:

```
GEMINI_API_KEY=your_actual_api_key_here
```

> **Never** commit `.env` to version control. It is already listed in `.gitignore`.

---

## Adding PDFs

Place your facility PDF manuals and SOPs in the `data/documents/` folder:

```
data/
└── documents/
    ├── chiller_sop.pdf
    ├── hvac_manual.pdf
    └── fire_suppression_guide.pdf
```

Alternatively, upload them directly through the Streamlit sidebar.

---

## Building the Knowledge Base

### Option A – CLI (recommended for large document sets)

```bash
python ingest.py
```

### Option B – Streamlit UI

1. Open the sidebar.
2. Upload PDFs using the file uploader.
3. Click **Build Knowledge Base**.

Both methods produce:

```
vectorstore/
├── index.faiss
└── metadata.pkl
```

Rebuild whenever you add, remove, or update documents.

---

## Running the Application

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## Example Questions

```
"The chiller on Floor 3 keeps tripping on high pressure. What should I check?"

"What is the recommended condenser water flow rate for the main chiller?"

"Outline the emergency shutdown procedure for the AHU."

"What maintenance is required for the cooling tower every quarter?"

"My BMS is showing fault code E-47. What does that mean?"
```

---

## How RAG Works

1. **Ingestion**: Each PDF is read page-by-page with PyMuPDF. Pages are split into 800-character chunks with 100-character overlap. Each chunk is embedded with `all-MiniLM-L6-v2` (384-dimensional vectors) and stored in a FAISS flat index.

2. **Retrieval**: Your question is embedded with the same model. FAISS performs cosine similarity search (via inner product on normalised vectors) and returns the top 5 most relevant chunks.

3. **Generation**: The 5 chunks (with source and page metadata) are injected into a strict system prompt and sent to Gemini. Gemini is instructed to answer **only** from that context.

4. **Citation**: Each answer is accompanied by the source PDF filename and page number drawn directly from the retrieved metadata — no fabrication.

---

## Configuration

All tunable values live in [`src/config.py`](src/config.py):

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | 5 | Chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |

---

## Limitations

- **Text-only PDFs**: Scanned documents (image-based PDFs) are not supported. The PDF must contain selectable text.
- **English-centric**: `all-MiniLM-L6-v2` works best with English-language documents.
- **No authentication**: This is a local MVP — no login or access control.
- **Single index**: All documents share one FAISS index. Rebuilding replaces everything.
- **Context window**: Very long answers requiring many pages of context may be truncated by Gemini's input limit.

---

## Future Scope

- OCR support for scanned PDFs (e.g., Tesseract)
- Multi-language embedding models
- Hybrid search (keyword + semantic)
- Per-user document collections
- Scheduled automatic re-ingestion
- Export chat logs to PDF
- Equipment fault-code lookup table
- Integration with BMS/CMMS APIs
