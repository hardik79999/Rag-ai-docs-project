from fastapi import APIRouter, HTTPException

from models.schemas import QueryRequest, QueryResponse
from services.rag_service import answer_question

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """POST /query — Ask a question; returns AI answer + source chunks."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question empty hai")

    result = answer_question(req.question, req.doc_ids)
    return result
