"""SQLAlchemy ORM models.

These are used when persisting document metadata to a relational DB.
For the default ChromaDB setup the in-memory dict in main.py is sufficient,
but these models are ready to drop in when you add PostgreSQL.
"""

from sqlalchemy import Column, DateTime, Integer, String, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Metadata for each uploaded PDF."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String(8), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    total_pages = Column(Integer, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
