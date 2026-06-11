"""
Universal File Parser
---------------------
Extracts clean text from any supported file type and returns
the same page-dict format used by pdf_service.py so the rest
of the pipeline (chunker → embedder → vector_store) works unchanged.

Supported formats:
  PDF   — PyMuPDF
  DOCX  — python-docx
  XLSX  — openpyxl
  PPTX  — python-pptx
  TXT   — plain text
  MD    — Markdown as plain text
  CSV   — comma-separated values
  HTML  — BeautifulSoup
  SVG   — extract text elements (diagrams with labels)
  Image (PNG/JPG/WEBP) — pytesseract OCR (optional, graceful fallback)
"""

import csv
import io
import re
from pathlib import Path


# ── Dispatcher ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc",
    ".xlsx", ".xls",
    ".pptx", ".ppt",
    ".txt", ".md", ".csv",
    ".html", ".htm",
    ".svg",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
}


def extract_file(file_path: str) -> dict:
    """
    Auto-detect file type and extract text.

    Returns:
        {
            "total_pages": int,
            "pages": [{"page_number": int, "text": str}, ...],
            "title": str,
        }
    Raises:
        ValueError  — unsupported extension
        Exception   — parsing failure
    """
    ext = Path(file_path).suffix.lower()

    handlers = {
        ".pdf":                          _extract_pdf,
        ".docx": _extract_docx, ".doc": _extract_docx,
        ".xlsx": _extract_xlsx, ".xls": _extract_xlsx,
        ".pptx": _extract_pptx, ".ppt": _extract_pptx,
        ".txt":  _extract_txt,  ".md":  _extract_txt,
        ".csv":  _extract_csv,
        ".html": _extract_html, ".htm": _extract_html,
        ".svg":  _extract_svg,
        ".png":  _extract_image, ".jpg": _extract_image,
        ".jpeg": _extract_image, ".webp": _extract_image,
        ".bmp":  _extract_image, ".tiff": _extract_image,
    }

    handler = handlers.get(ext)
    if not handler:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    result = handler(file_path)
    result.setdefault("title", Path(file_path).stem)
    return result


# ── PDF ───────────────────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> dict:
    from services.pdf_service import extract_text_from_pdf
    return extract_text_from_pdf(path)


# ── DOCX ──────────────────────────────────────────────────────────────────────

def _extract_docx(path: str) -> dict:
    import docx
    doc = docx.Document(path)

    pages, current_parts, page_num = [], [], 1
    current_heading = ""

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = para.style.name.lower()
        if "heading" in style:
            # flush current section
            if current_parts:
                pages.append({
                    "page_number": page_num,
                    "text": (f"{current_heading}\n\n" if current_heading else "") + " ".join(current_parts)
                })
                page_num += 1
                current_parts = []
            current_heading = text
        else:
            current_parts.append(text)

    # flush last section
    if current_parts:
        pages.append({
            "page_number": page_num,
            "text": (f"{current_heading}\n\n" if current_heading else "") + " ".join(current_parts)
        })

    # Also extract tables
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            pages.append({"page_number": page_num, "text": "\n".join(rows)})
            page_num += 1

    if not pages:
        raise ValueError("No text found in DOCX file")

    return {"total_pages": len(pages), "pages": pages}


# ── XLSX ──────────────────────────────────────────────────────────────────────

def _extract_xlsx(path: str) -> dict:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    pages = []

    for sheet_num, sheet in enumerate(wb.worksheets, 1):
        rows = []
        headers = []
        for r_idx, row in enumerate(sheet.iter_rows(values_only=True)):
            cells = [str(c).strip() if c is not None else "" for c in row]
            if not any(cells):
                continue
            if r_idx == 0:
                headers = cells
                rows.append("  |  ".join(cells))  # header row
            else:
                if headers:
                    pairs = [f"{h}: {v}" for h, v in zip(headers, cells) if v]
                    rows.append("  |  ".join(pairs) if pairs else "  |  ".join(cells))
                else:
                    rows.append("  |  ".join(c for c in cells if c))

        if rows:
            pages.append({
                "page_number": sheet_num,
                "text": f"Sheet: {sheet.title}\n\n" + "\n".join(rows)
            })

    wb.close()
    if not pages:
        raise ValueError("No data found in Excel file")
    return {"total_pages": len(pages), "pages": pages}


# ── PPTX ──────────────────────────────────────────────────────────────────────

def _extract_pptx(path: str) -> dict:
    from pptx import Presentation
    prs = Presentation(path)
    pages = []

    for slide_num, slide in enumerate(prs.slides, 1):
        parts = []
        title_text = ""

        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                text = " ".join(run.text for run in para.runs).strip()
                if not text:
                    continue
                # detect title shape
                if shape.shape_type == 13 or (hasattr(shape, "placeholder_format") and
                   shape.placeholder_format and shape.placeholder_format.idx == 0):
                    title_text = text
                else:
                    parts.append(text)

        full = (f"{title_text}\n\n" if title_text else "") + "\n".join(parts)
        if full.strip():
            pages.append({"page_number": slide_num, "text": full.strip()})

    if not pages:
        raise ValueError("No text found in PowerPoint file")
    return {"total_pages": len(pages), "pages": pages}


# ── TXT / MD ──────────────────────────────────────────────────────────────────

def _extract_txt(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Split into ~1500-char logical sections by blank lines / headings
    sections = re.split(r"\n{3,}|(?=^#{1,3} )", text, flags=re.MULTILINE)
    pages = []
    for i, s in enumerate(sections, 1):
        s = s.strip()
        if len(s) > 40:
            pages.append({"page_number": i, "text": s})

    if not pages:
        pages = [{"page_number": 1, "text": text.strip()}]

    return {"total_pages": len(pages), "pages": pages}


# ── CSV ───────────────────────────────────────────────────────────────────────

def _extract_csv(path: str) -> dict:
    rows_text = []
    headers = []

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for r_idx, row in enumerate(reader):
            if r_idx == 0:
                headers = row
                rows_text.append("  |  ".join(row))
            else:
                if headers:
                    pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v.strip()]
                    rows_text.append("  |  ".join(pairs) if pairs else "  |  ".join(row))
                else:
                    rows_text.append("  |  ".join(row))

    if not rows_text:
        raise ValueError("CSV file is empty")

    # Group into pages of 100 rows each
    chunk_size = 100
    pages = []
    for i in range(0, len(rows_text), chunk_size):
        chunk = rows_text[i:i + chunk_size]
        pages.append({
            "page_number": i // chunk_size + 1,
            "text": "\n".join(chunk)
        })

    return {"total_pages": len(pages), "pages": pages}


# ── HTML ──────────────────────────────────────────────────────────────────────

def _extract_html(path: str) -> dict:
    from services.url_service import _parse_html
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    result = _parse_html(html, path)
    return result


# ── SVG ───────────────────────────────────────────────────────────────────────

def _extract_svg(path: str) -> dict:
    from bs4 import BeautifulSoup
    svg = Path(path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(svg, "lxml")

    texts = []
    for el in soup.find_all(["text", "tspan", "title", "desc"]):
        t = el.get_text(strip=True)
        if t and len(t) > 1:
            texts.append(t)

    if not texts:
        raise ValueError("No readable text found in SVG file")

    content = "\n".join(texts)
    return {
        "total_pages": 1,
        "pages": [{"page_number": 1, "text": content}]
    }


# ── Image (OCR) ───────────────────────────────────────────────────────────────

def _extract_image(path: str) -> dict:
    try:
        import pytesseract
        from PIL import Image
        img  = Image.open(path)
        text = pytesseract.image_to_string(img).strip()
        if not text:
            raise ValueError("No text detected in image (OCR returned empty)")
        return {
            "total_pages": 1,
            "pages": [{"page_number": 1, "text": text}]
        }
    except ImportError:
        raise ValueError(
            "Image OCR requires pytesseract and tesseract-ocr to be installed.\n"
            "Run: sudo apt install tesseract-ocr && pip install pytesseract Pillow"
        )
