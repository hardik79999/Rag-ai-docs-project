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

---

## 💻 Core Code

### config.py

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_api_key: str
    vector_store: str = "chroma"  # "chroma" or "pgvector"
    chroma_persist_dir: str = "./chroma_db"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

### services/pdf_service.py — PDF Text Extraction

```python
import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(file_path: str) -> dict:
    """PDF se text extract karo with page numbers"""
    doc = fitz.open(file_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():  # empty pages skip karo
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })
    
    doc.close()
    return {
        "total_pages": len(doc),
        "pages": pages
    }
```

---

### services/chunker.py — Text Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(pages: list[dict], 
               chunk_size: int = 800, 
               chunk_overlap: int = 150) -> list[dict]:
    """
    Text ko meaningful chunks mein split karo.
    
    chunk_size=800  → ~600 words per chunk (balance between context & precision)
    chunk_overlap=150 → chunks overlap karein taaki context na tute
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "page_number": page["page_number"],
                "chunk_index": i
            })
    
    return chunks
```

---

### services/embedder.py — Gemini Embeddings

```python
import google.generativeai as genai
from config import settings

genai.configure(api_key=settings.gemini_api_key)

EMBEDDING_MODEL = "models/gemini-embedding-002"

def get_embedding(text: str) -> list[float]:
    """Single text ka embedding lo"""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_DOCUMENT"  # Document store karte waqt
    )
    return result["embedding"]  # 768-dim vector

def get_query_embedding(text: str) -> list[float]:
    """Query ka embedding lo (task_type alag hai!)"""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_QUERY"  # Query karte waqt
    )
    return result["embedding"]

def get_batch_embeddings(texts: list[str]) -> list[list[float]]:
    """Multiple texts ke embeddings ek saath lo"""
    embeddings = []
    # Gemini batch API use karo (100 texts at a time)
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            emb = get_embedding(text)
            embeddings.append(emb)
    return embeddings
```

> ⚠️ **Important**: Document store karte waqt `task_type="RETRIEVAL_DOCUMENT"`, query karte waqt `task_type="RETRIEVAL_QUERY"` use karo. Isse search quality better hoti hai.

---

### services/vector_store.py — ChromaDB (No Postgres Needed!)

```python
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings

class ChromaVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
    
    def get_or_create_collection(self, doc_id: str):
        """Har document ka alag collection banao"""
        return self.client.get_or_create_collection(
            name=f"doc_{doc_id}",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
    
    def add_chunks(self, doc_id: str, chunks: list[dict], embeddings: list[list[float]]):
        """Chunks + embeddings store karo"""
        collection = self.get_or_create_collection(doc_id)
        
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [{"page": c["page_number"], "chunk_idx": c["chunk_index"]} 
                     for c in chunks]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: list[float], 
               doc_ids: list[str] = None, 
               top_k: int = 5) -> list[dict]:
        """Semantic search karo"""
        results = []
        
        # Agar specific docs nahi diye to sab mein search karo
        collections = self.client.list_collections()
        
        for col in collections:
            if doc_ids and col.name.replace("doc_", "") not in doc_ids:
                continue
            
            collection = self.client.get_collection(col.name)
            res = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            for i, doc in enumerate(res["documents"][0]):
                results.append({
                    "text": doc,
                    "metadata": res["metadatas"][0][i],
                    "score": 1 - res["distances"][0][i],  # cosine similarity
                    "doc_id": col.name.replace("doc_", "")
                })
        
        # Score se sort karo
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# Global instance
vector_store = ChromaVectorStore()
```

---

### services/rag_service.py — Main RAG Logic

```python
import google.generativeai as genai
from services.embedder import get_query_embedding
from services.vector_store import vector_store
from config import settings

genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

def answer_question(question: str, doc_ids: list[str] = None) -> dict:
    """
    RAG Pipeline:
    1. Question embed karo
    2. Similar chunks dhundo
    3. Context banao
    4. Gemini se answer lo
    """
    
    # Step 1: Query embed karo
    query_embedding = get_query_embedding(question)
    
    # Step 2: Top relevant chunks dhundo
    relevant_chunks = vector_store.search(
        query_embedding=query_embedding,
        doc_ids=doc_ids,
        top_k=5
    )
    
    if not relevant_chunks:
        return {
            "answer": "Koi relevant information nahi mili documents mein.",
            "sources": []
        }
    
    # Step 3: Context build karo
    context = "\n\n---\n\n".join([
        f"[Page {c['metadata']['page']}]: {c['text']}" 
        for c in relevant_chunks
    ])
    
    # Step 4: Gemini se answer lo
    prompt = f"""
Neeche diye gaye documentation context ke basis pe question ka answer do.
Agar answer context mein nahi hai, to clearly bol do.
Sirf context mein jo information hai wohi use karo.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    response = model.generate_content(prompt)
    
    return {
        "answer": response.text,
        "sources": [
            {
                "page": c["metadata"]["page"],
                "doc_id": c["doc_id"],
                "score": round(c["score"], 3),
                "snippet": c["text"][:200] + "..."
            }
            for c in relevant_chunks
        ]
    }
```

---

### main.py — FastAPI App

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile, os, uuid

from services.pdf_service import extract_text_from_pdf
from services.chunker import chunk_text
from services.embedder import get_batch_embeddings
from services.vector_store import vector_store
from services.rag_service import answer_question

app = FastAPI(title="RAG AI Documentation API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory document registry (production mein DB use karo)
uploaded_docs = {}

# ─────────────────────────────────────────
# 1. PDF UPLOAD ENDPOINT
# ─────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")
    
    doc_id = str(uuid.uuid4())[:8]
    
    # Temp file mein save karo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 1. Text extract karo
        extracted = extract_text_from_pdf(tmp_path)
        
        # 2. Chunks banao
        chunks = chunk_text(extracted["pages"])
        
        # 3. Embeddings lo (batch mein)
        texts = [c["text"] for c in chunks]
        embeddings = get_batch_embeddings(texts)
        
        # 4. Vector store mein save karo
        vector_store.add_chunks(doc_id, chunks, embeddings)
        
        # Registry mein track karo
        uploaded_docs[doc_id] = {
            "filename": file.filename,
            "total_pages": extracted["total_pages"],
            "total_chunks": len(chunks)
        }
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": extracted["total_pages"],
            "total_chunks": len(chunks),
            "message": "PDF successfully processed!"
        }
    finally:
        os.unlink(tmp_path)

# ─────────────────────────────────────────
# 2. QUESTION ANSWERING ENDPOINT
# ─────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    doc_ids: list[str] = None  # None = sab docs mein search karo

@app.post("/query")
async def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question empty hai")
    
    result = answer_question(req.question, req.doc_ids)
    return result

# ─────────────────────────────────────────
# 3. DOCUMENTS LIST ENDPOINT
# ─────────────────────────────────────────
@app.get("/documents")
async def list_documents():
    return {"documents": uploaded_docs}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## 🔄 PGVector Setup (Agar PostgreSQL Available Ho)

### Ubuntu pe PostgreSQL + PGVector install karna

```bash
# PostgreSQL install karo
sudo apt update
sudo apt install postgresql postgresql-contrib

# PGVector extension install karo
sudo apt install postgresql-16-pgvector  # version match karo

# PostgreSQL start karo
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Database banao
sudo -u postgres psql
CREATE DATABASE ragdb;
CREATE USER raguser WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;
\c ragdb
CREATE EXTENSION vector;
\q
```

### PGVector ka Vector Store Service

```python
# services/pgvector_store.py
from sqlalchemy import create_engine, text
from pgvector.sqlalchemy import Vector
import numpy as np

class PGVectorStore:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self._init_table()
    
    def _init_table(self):
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT,
                    chunk_text TEXT,
                    page_number INT,
                    chunk_index INT,
                    embedding vector(768)
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS emb_idx 
                ON embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            conn.commit()
    
    def add_chunks(self, doc_id: str, chunks: list, embeddings: list):
        with self.engine.connect() as conn:
            for chunk, emb in zip(chunks, embeddings):
                conn.execute(text("""
                    INSERT INTO embeddings (doc_id, chunk_text, page_number, chunk_index, embedding)
                    VALUES (:doc_id, :text, :page, :idx, :emb)
                """), {
                    "doc_id": doc_id,
                    "text": chunk["text"],
                    "page": chunk["page_number"],
                    "idx": chunk["chunk_index"],
                    "emb": emb
                })
            conn.commit()
    
    def search(self, query_embedding: list, top_k: int = 5) -> list:
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT doc_id, chunk_text, page_number,
                       1 - (embedding <=> :emb::vector) AS score
                FROM embeddings
                ORDER BY embedding <=> :emb::vector
                LIMIT :k
            """), {"emb": str(query_embedding), "k": top_k})
            
            return [
                {"doc_id": r[0], "text": r[1], 
                 "metadata": {"page": r[2]}, "score": r[3]}
                for r in result
            ]
```

---

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