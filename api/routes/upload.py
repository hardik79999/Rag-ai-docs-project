import tempfile
import os
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.chunker import chunk_text
from services.embedder import get_batch_embeddings
from services.pdf_service import extract_text_from_pdf
from services.vector_store import vector_store

router = APIRouter()

# Shared in-memory registry (import this in main.py if needed)
uploaded_docs: dict = {}


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """POST /upload — Accept a PDF, extract text, chunk it, embed and store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    doc_id = str(uuid.uuid4())[:8]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        extracted = extract_text_from_pdf(tmp_path)
        chunks = chunk_text(extracted["pages"])
        texts = [c["text"] for c in chunks]
        embeddings = get_batch_embeddings(texts)
        vector_store.add_chunks(doc_id, chunks, embeddings)

        uploaded_docs[doc_id] = {
            "filename": file.filename,
            "total_pages": extracted["total_pages"],
            "total_chunks": len(chunks),
        }

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": extracted["total_pages"],
            "total_chunks": len(chunks),
            "message": "PDF successfully processed!",
        }
    finally:
        os.unlink(tmp_path)
