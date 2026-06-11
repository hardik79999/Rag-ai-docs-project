# FIXED:
# 1. /query wrapped in broad try/except → HTTP 429 or 500, never crashes
# 2. /upload wrapped in try/except → HTTP 500 with detail on embedding failures
# 3. NEW: POST /upload/bulk — accepts multiple PDFs in one request
# 4. Duplicate imports removed

import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.chunker import chunk_text
from services.embedder import get_batch_embeddings
from services.file_service import extract_file, SUPPORTED_EXTENSIONS
from services.url_service import fetch_url
from services.rag_service import answer_question
from services.vector_store import vector_store

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG AI Documentation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document registry
uploaded_docs: dict = {}


# ─────────────────────────────────────────
# HELPER — process any supported file
# ─────────────────────────────────────────
async def _process_file(file: UploadFile) -> dict:
    """Extract, chunk, embed and store any supported file. Returns metadata dict."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    doc_id = str(uuid.uuid4())[:8]

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        extracted  = extract_file(tmp_path)
        chunks     = chunk_text(extracted["pages"])
        if not chunks:
            raise ValueError("No extractable text found in file")
        
        # SAFEGUARD: Limit to 2000 chunks (approx 1.5 - 2MB of text) 
        # to prevent hitting the 100 requests/minute free-tier Gemini limit.
        warning_msg = None
        if len(chunks) > 2000:
            logger.warning(f"File {file.filename} too large. Truncating from {len(chunks)} to 2000 chunks.")
            chunks = chunks[:2000]
            warning_msg = "File too large for Free Tier. Only the first ~2MB were indexed."

        texts      = [c["text"] for c in chunks]
        embeddings = get_batch_embeddings(texts)
        vector_store.add_chunks(doc_id, chunks, embeddings)

        meta = {
            "doc_id":       doc_id,
            "filename":     file.filename,
            "file_type":    ext.lstrip(".").upper(),
            "total_pages":  extracted["total_pages"],
            "total_chunks": len(chunks),
            "message":      warning_msg or "File successfully processed!",
        }
        uploaded_docs[doc_id] = {
            "filename":     file.filename,
            "file_type":    ext.lstrip(".").upper(),
            "total_pages":  extracted["total_pages"],
            "total_chunks": len(chunks),
        }
        return meta
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────
# 1. SINGLE FILE UPLOAD (any format)
# ─────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """POST /upload — process one file (PDF, DOCX, XLSX, PPTX, TXT, CSV, HTML, SVG, Image)."""
    try:
        return await _process_file(file)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ─────────────────────────────────────────
# 2. BULK FILE UPLOAD (any format, mixed)
# ─────────────────────────────────────────
@app.post("/upload/bulk")
async def upload_bulk(files: list[UploadFile] = File(...)):
    """POST /upload/bulk — process multiple files of any supported type."""
    if not files:
        raise HTTPException(400, "No files provided")

    results = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status":   "skipped",
                "error":    f"Unsupported type '{ext}'",
            })
            continue
        try:
            meta = await _process_file(file)
            results.append({"status": "ok", **meta})
        except Exception as e:
            logger.error(f"Bulk upload failed for {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status":   "error",
                "error":    str(e),
            })

    ok_count  = sum(1 for r in results if r["status"] == "ok")
    err_count = len(results) - ok_count
    return {
        "summary": {"total": len(results), "success": ok_count, "failed": err_count},
        "results": results,
    }


@app.get("/supported-formats")
async def supported_formats():
    """GET /supported-formats — list all supported file extensions."""
    return {"extensions": sorted(SUPPORTED_EXTENSIONS)}


# ─────────────────────────────────────────
# 3. INGEST URL  (NEW)
# ─────────────────────────────────────────
class IngestUrlRequest(BaseModel):
    url: str


@app.post("/ingest-url")
async def ingest_url(req: IngestUrlRequest):
    """POST /ingest-url — scrape a URL, chunk, embed and store like a PDF."""
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL cannot be empty")

    try:
        # 1. Fetch + extract text (same shape as pdf_service output)
        extracted = fetch_url(url)

        # 2. Chunk
        chunks = chunk_text(extracted["pages"])
        if not chunks:
            raise HTTPException(422, "No extractable text found at this URL")

        # 3. Embed
        texts      = [c["text"] for c in chunks]
        embeddings = get_batch_embeddings(texts)

        # 4. Store — use doc_id based on domain so re-ingesting same site updates it
        from urllib.parse import urlparse
        domain  = urlparse(url).netloc.replace("www.", "")
        doc_id  = str(uuid.uuid4())[:8]
        display = extracted.get("title") or domain or url[:40]

        vector_store.add_chunks(doc_id, chunks, embeddings)

        uploaded_docs[doc_id] = {
            "filename":     display,
            "source_url":   url,
            "total_pages":  extracted["total_pages"],
            "total_chunks": len(chunks),
            "type":         "url",
        }

        return {
            "doc_id":       doc_id,
            "title":        display,
            "url":          url,
            "total_pages":  extracted["total_pages"],
            "total_chunks": len(chunks),
            "message":      "URL successfully ingested!",
        }

    except HTTPException:
        raise
    except Exception as e:
        err = str(e)
        logger.error(f"URL ingest failed for {url}: {err}")
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            raise HTTPException(429, "Gemini quota exhausted during embedding")
        raise HTTPException(500, f"Failed to ingest URL: {err}")


# ─────────────────────────────────────────
# 4. QUESTION ANSWERING
# ─────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None          # None = search all documents
    chat_history: list[dict] | None = None    # [{"role":"user"|"ai","content":"..."}]


@app.post("/query")
async def query_documents(req: QueryRequest):
    """POST /query — ask a question; returns AI answer + query type + source chunks."""
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    try:
        result = answer_question(req.question, req.doc_ids, req.chat_history)
        return result

    except Exception as e:
        err_str = str(e)
        logger.error(f"Query failed: {err_str}")

        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini quota exhausted. Wait for the daily reset "
                    "(midnight UTC) or enable billing at "
                    "https://aistudio.google.com"
                ),
            )

        raise HTTPException(status_code=500, detail=f"Query failed: {err_str}")


# ─────────────────────────────────────────
# 5. DOCUMENTS LIST
# ─────────────────────────────────────────
@app.get("/documents")
async def list_documents():
    return {"documents": uploaded_docs}


# ─────────────────────────────────────────
# 6. HEALTH CHECK
# ─────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ─────────────────────────────────────────
# 7. SERVE FRONTEND
# ─────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    @app.get("/")
    async def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
