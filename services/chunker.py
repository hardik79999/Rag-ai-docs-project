from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """Text ko meaningful chunks mein split karo.

    Args:
        pages: List of {"page_number": int, "text": str} dicts.
        chunk_size: Max characters per chunk (~600 words).
        chunk_overlap: Overlap between consecutive chunks to preserve context.

    Returns:
        List of {"text": str, "page_number": int, "chunk_index": int} dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "page_number": page["page_number"],
                "chunk_index": i,
            })

    return chunks
