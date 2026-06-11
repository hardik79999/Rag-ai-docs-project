"""Vector store abstraction — ChromaDB (default) or PGVector.

Set VECTOR_STORE=chroma  (default) or VECTOR_STORE=pgvector in .env.
"""

import chromadb

from config import settings


class ChromaVectorStore:
    """Local persistent vector store backed by ChromaDB."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    def get_or_create_collection(self, doc_id: str):
        """Each document gets its own Chroma collection."""
        return self.client.get_or_create_collection(
            name=f"doc_{doc_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks + pre-computed embeddings."""
        collection = self.get_or_create_collection(doc_id)

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {"page": c["page_number"], "chunk_idx": c["chunk_index"]}
            for c in chunks
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        doc_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Semantic search across all (or specific) document collections."""
        results = []
        collections = self.client.list_collections()

        for col in collections:
            col_doc_id = col.name.replace("doc_", "")
            if doc_ids and col_doc_id not in doc_ids:
                continue

            collection = self.client.get_collection(col.name)
            count = collection.count()
            if count == 0:
                continue

            res = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )

            for i, doc in enumerate(res["documents"][0]):
                results.append({
                    "text": doc,
                    "metadata": res["metadatas"][0][i],
                    "score": 1 - res["distances"][0][i],  # cosine similarity
                    "doc_id": col_doc_id,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def _build_vector_store():
    """Factory: return the correct store based on VECTOR_STORE env var."""
    if settings.vector_store == "pgvector":
        # Lazy import so ChromaDB users don't need pgvector installed
        from services.pgvector_store import PGVectorStore  # noqa: PLC0415
        return PGVectorStore(settings.database_url)
    return ChromaVectorStore()


# Module-level singleton
vector_store = _build_vector_store()
