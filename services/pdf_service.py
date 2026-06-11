import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> dict:
    """PDF se text extract karo with page numbers.

    Returns:
        {
            "total_pages": int,
            "pages": [{"page_number": int, "text": str}, ...]
        }
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():  # empty pages skip karo
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip(),
            })

    total = len(doc)
    doc.close()

    return {
        "total_pages": total,
        "pages": pages,
    }
