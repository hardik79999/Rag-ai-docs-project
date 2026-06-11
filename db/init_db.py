"""Create all tables and enable the pgvector extension.

Run once before starting the server when using PostgreSQL:
    python db/init_db.py
"""

from sqlalchemy import text

from db.database import engine
from models.document import Base


def init_db() -> None:
    with engine.connect() as conn:
        # Enable pgvector extension (no-op if already enabled)
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create all SQLAlchemy-mapped tables
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialised — tables created and pgvector extension enabled.")


if __name__ == "__main__":
    init_db()
