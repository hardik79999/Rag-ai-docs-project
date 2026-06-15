# Rag-ai-docs-project

# 📚 RAG AI Documentation System
## FastAPI + Gemini Embeddings + PGVector (with ChromaDB fallback)

---

## 🧠 Project Overview

Ek system banayenge jisme:
- **Bulk PDF & URL ingest** karo → automatically chunks + embed hoga
- **Question pucho** → AI semantic search karega → relevant context dhundega → Gemini se answer milega
- **Technology**: FastAPI + Gemini Embedding + PGVector (ya ChromaDB if no Postgres)

✨ **Key Optimizations Included:**
- ⚡ **Fully Asynchronous & Non-Blocking**: Thread-pooling for CPU-bound tasks and async Gemini API calls.
- 🎯 **Hybrid Search**: Dense Vector Search (Cosine) + Keyword Overlap Boost (TF) for pinpoint accuracy.
- 🚀 **Semantic Caching**: In-memory `TTLCache` to instantly serve identical queries and save API quota.

---

## 🏗️ Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
│                                                             │
│  PDF Files  →  Extract Text  →  Chunking  →  Embedding     │
│                (PyMuPDF)         (LangChain)   (Gemini)     │
│                                                   ↓         │
│                                            Vector Store      │
│                                         (PGVector/Chroma)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                          │
│                                                             │
│  User Question  →  Check Cache  →  Embed Query  →           │
│      ↓               (TTLCache)      (Gemini)               │
│  Vector Search + Keyword Boost  →  Top K Chunks             │
│  (Hybrid Score)                       ↓                     │
│                              Context + Question → Gemini    │
│                                                    ↓        │
│                                              Final Answer   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | REST endpoints |
| **PDF Parsing** | PyMuPDF (fitz) | Text extract from PDF |
| **Chunking** | LangChain Text Splitter | Split text into chunks |
| **Embedding Model** | `gemini-embedding-001` | Text → Vector (768 dim) |
| **LLM** | `gemini-2.0-flash-lite` | Question answering |
| **Vector DB (Option A)** | PGVector (PostgreSQL) | Production use |
| **Vector DB (Option B)** | ChromaDB | Local, no Postgres needed ✅ |
| **Caching** | `SimpleTTLCache` (Custom) | Caches semantic queries for 1 hr |
| **Async Framework** | `asyncio` & `google-genai aio` | Non-blocking API requests |

---

## 📁 Project Structure

```
rag-ai-docs/
├── main.py                  # FastAPI app entry point
├── requirements.txt
├── .env                     # API keys
├── config.py                # Settings
│
├── api/
│   ├── routes/
│   │   ├── upload.py        # POST /upload - PDF upload endpoint
│   │   ├── query.py         # POST /query - Ask question
│   │   └── documents.py     # GET /documents - list uploaded docs
│
├── services/
│   ├── pdf_service.py       # PDF text extraction
│   ├── chunker.py           # Text chunking logic
│   ├── embedder.py          # Gemini embedding calls
│   ├── vector_store.py      # PGVector / ChromaDB abstraction
│   └── rag_service.py       # Main RAG orchestration
│
├── models/
│   ├── document.py          # SQLAlchemy models
│   └── schemas.py           # Pydantic schemas
│
└── db/
    ├── database.py          # DB connection
    └── init_db.py           # Create tables / PGVector extension
```

---

## ⚙️ Installation & Setup

### 1. Python Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# OR
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 2. requirements.txt

```txt
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9       # File upload support
pymupdf==1.24.0               # PDF parsing (fitz)
langchain-text-splitters==0.2.0
google-generativeai==0.7.0    # Gemini API
chromadb==0.5.0               # Local vector store (no Postgres needed)
sqlalchemy==2.0.30
python-dotenv==1.0.1
httpx==0.27.0
pydantic==2.7.0

# Only if using PGVector:
# psycopg2-binary==2.9.9
# asyncpg==0.29.0
# pgvector==0.2.5
```

### 3. .env File

```env
GEMINI_API_KEY=your_gemini_api_key_here

# Only if using PostgreSQL + PGVector:
# DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ragdb

# Vector store choice: "chroma" or "pgvector"
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=./chroma_db
```

---

## 🔑 Gemini API Key Kaise Le (AI Studio)

1. **https://aistudio.google.com** pe jao
2. **"Get API Key"** click karo
3. **"Create API key"** → copy karo
4. `.env` file mein `GEMINI_API_KEY=...` set karo

**Models jo use karenge:**
- Embedding: `models/gemini-embedding-002` → 768-dimensional vectors
- LLM: `models/gemini-2.0-flash-lite` → fast, cheap, good quality



## 🚀 Run Kaise Karo

```bash
# Development server start karo
uvicorn main:app --reload --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**API Docs**: http://localhost:8000/docs (Swagger UI auto-generate hoti hai!)

---

## 📬 API Usage Examples

### PDF Upload
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@my_documentation.pdf"

# Response:
# {
#   "doc_id": "a1b2c3d4",
#   "filename": "my_documentation.pdf",
#   "total_pages": 42,
#   "total_chunks": 156,
#   "message": "PDF successfully processed!"
# }
```

### Question Pucho
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "FastAPI mein authentication kaise karte hain?"}'

# Response:
# {
#   "answer": "FastAPI mein authentication ke liye...",
#   "sources": [
#     {"page": 12, "doc_id": "a1b2c3d4", "score": 0.89, "snippet": "..."}
#   ]
# }
```

### Specific Document mein Search
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Database connection setup?", "doc_ids": ["a1b2c3d4"]}'
```

---

## 🧩 Chunking Strategy Explained

```
Original PDF Page Text (2000 words)
           ↓
RecursiveCharacterTextSplitter
           ↓
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Chunk 1  │  │ Chunk 2  │  │ Chunk 3  │
│ 800 chars│  │ 800 chars│  │ 800 chars│
└──────────┘  └──────────┘  └──────────┘
         ↑──150──↑     ↑──150──↑
              (overlap)

chunk_size=800   → Context ke liye enough
chunk_overlap=150 → Chunk boundary pe information loss nahi
```

**Why RecursiveCharacterTextSplitter?**
- Pehle `\n\n` (paragraphs) pe split karta hai
- Phir `\n` (lines) pe
- Phir `. ` (sentences) pe
- Natural text boundaries respect karta hai

---

## 📊 Embedding Model Details

| Property | Value |
|----------|-------|
| **Model** | `gemini-embedding-002` |
| **Dimensions** | 768 |
| **Max Input Tokens** | 2048 |
| **task_type: RETRIEVAL_DOCUMENT** | Chunks store karte waqt |
| **task_type: RETRIEVAL_QUERY** | User query embed karte waqt |
| **Free Tier** | 1500 requests/day |

---

## 🔍 Semantic & Hybrid Search Flow (Visual)

```
User: "How to setup database connection?"
          ↓
    Check Cache: (Miss)
          ↓
    embed_query("How to setup database connection?")
          ↓
    [0.12, -0.45, 0.89, ... 768 numbers]
          ↓
    ChromaDB: cosine similarity with all stored chunks
          ↓
    Apply Hybrid Keyword Boost (check chunks for "database", "connection")
          ↓
    Top 5 most similar chunks (boosted):
    - Score: 0.91 (+0.05 boost) → "Database configuration in FastAPI..."
    - Score: 0.87 (+0.02 boost) → "SQLAlchemy connection setup..."
    - Score: 0.82 (+0.00 boost) → "Environment variables for DB..."
          ↓
    Context + Question → Gemini Flash Lite (Async)
          ↓
    "FastAPI mein database setup ke liye..." (Cached for 1 hour)
```

---

## ⚡ Quick Decision Guide

```
Mere paas PostgreSQL nahi hai?
    → VECTOR_STORE=chroma in .env
    → ChromaDB local files mein save karega (./chroma_db/)
    → Zero extra setup!

Production deploy karna hai?
    → PostgreSQL + PGVector setup karo
    → Better performance, concurrent users support
    → VECTOR_STORE=pgvector in .env

Free mein Gemini use karna hai?
    → Google AI Studio se API key lo
    → Free tier: 1500 embed requests/day, 1500 generate requests/day
    → Kafi hai development ke liye!
```

---

## 🐛 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `RESOURCE_EXHAUSTED` | Rate limit hit hua, 1 min wait karo |
| `Invalid API key` | .env mein key check karo |
| PDF text extract nahi ho raha | Scanned PDF hai? OCR use karna padega (pytesseract) |
| Chroma directory permission error | `chmod 755 ./chroma_db` |
| Embedding dimension mismatch | Purana collection delete karo, fresh start karo |

---

*Project by: RAG AI Documentation System | Stack: FastAPI + Gemini + ChromaDB/PGVector*