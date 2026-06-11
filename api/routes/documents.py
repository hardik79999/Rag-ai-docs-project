from fastapi import APIRouter

from api.routes.upload import uploaded_docs

router = APIRouter()


@router.get("/documents")
async def list_documents():
    """GET /documents — Return metadata for all uploaded documents."""
    return {"documents": uploaded_docs}
