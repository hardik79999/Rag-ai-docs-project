"""PGVector-backed vector store (optional — requires PostgreSQL + pgvector).

Only used when VECTOR_STORE=pgvector in .env.
Install extras: psycopg2-binary, asyncpg, pgvector
"""

from sqlalchemy import create_engine, text


class PGVectorStore:
    """Production-grade vector store using PostgreSQL + pgvector extension."""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self._init_table()

    def _init_table(self) -> None:
        """Create the embeddings table and HNSW index if they don't exist."""
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id          SERIAL PRIMARY KEY,
                    doc_id      TEXT,
                    chunk_text  TEXT,
                    page_number INT,
                    chunk_index INT,
                    embedding   vector(768)
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS emb_idx
                ON embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            conn.commit()

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        with self.engine.connect() as conn:
            for chunk, emb in zip(chunks, embeddings):
                conn.execute(
                    text("""
                        INSERT INTO embeddings
                            (doc_id, chunk_text, page_number, chunk_index, embedding)
                        VALUES (:doc_id, :text, :page, :idx, :emb)
                    """),
                    {
                        "doc_id": doc_id,
                        "text": chunk["text"],
                        "page": chunk["page_number"],
                        "idx": chunk["chunk_index"],
                        "emb": str(emb),
                    },
                )
            conn.commit()

    def search(
        self,
        query_embedding: list[float],
        doc_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        filter_clause = ""
        params: dict = {"emb": str(query_embedding), "k": top_k}

        if doc_ids:
            placeholders = ", ".join(f":id{i}" for i in range(len(doc_ids)))
            filter_clause = f"WHERE doc_id IN ({placeholders})"
            for i, did in enumerate(doc_ids):
                params[f"id{i}"] = did

        with self.engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT doc_id, chunk_text, page_number,
                           1 - (embedding <=> :emb::vector) AS score
                    FROM embeddings
                    {filter_clause}
                    ORDER BY embedding <=> :emb::vector
                    LIMIT :k
                """),
                params,
            )

            return [
                {
                    "doc_id": row[0],
                    "text": row[1],
                    "metadata": {"page": row[2]},
                    "score": float(row[3]),
                }
                for row in result
            ]
