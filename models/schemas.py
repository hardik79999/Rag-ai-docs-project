"""Pydantic schemas for request / response validation."""

from pydantic import BaseModel, Field


# ── Request schemas ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    doc_ids: list[str] | None = Field(
        default=None,
        description="Optional list of doc IDs to restrict search. "
                    "Pass null / omit to search all documents.",
    )


# ── Response schemas ───────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    page: int
    doc_id: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    total_pages: int
    total_chunks: int
    message: str


class DocumentMeta(BaseModel):
    filename: str
    total_pages: int
    total_chunks: int


class DocumentsResponse(BaseModel):
    documents: dict[str, DocumentMeta]
