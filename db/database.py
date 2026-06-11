"""Database connection helpers (used when VECTOR_STORE=pgvector)."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import settings

# Synchronous engine — swap for create_async_engine + AsyncSession for async routes
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,   # reconnect on stale connections
    echo=False,           # set True to log SQL statements during development
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
