"""
URL Ingestion Service
---------------------
Scrapes a URL, extracts clean readable text, and returns it in the same
page-dict format that pdf_service.py uses — so the rest of the pipeline
(chunker → embedder → vector_store) works without any changes.

Supports:
  - Regular web pages (HTML)
  - Auto-detects PDF URLs → falls back to pdf_service

Smart extraction:
  - Removes nav, footer, ads, scripts, styles
  - Prefers <article> / <main> / largest content block
  - Splits by heading hierarchy to preserve document structure
"""

import re
import httpx
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse


# ── Constants ──────────────────────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_TIMEOUT = 20  # seconds
_SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header",
              "aside", "form", "button", "iframe", "svg", "meta", "link",
              "advertisement", "ads"}
_CONTENT_TAGS = {"article", "main", "section", "div", "p"}


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_url(url: str) -> dict:
    """
    Fetch a URL and extract clean text.

    Returns the same shape as pdf_service.extract_text_from_pdf():
    {
        "total_pages": int,       # number of logical "sections"
        "pages": [
            {"page_number": int, "text": str},
            ...
        ],
        "title": str,
        "url": str,
    }

    Raises:
        ValueError  — bad URL, non-HTML content (that isn't PDF)
        httpx.HTTPError — network / HTTP errors
    """
    url = url.strip()
    _validate_url(url)

    # PDF URL → delegate to pdf_service
    if url.lower().endswith(".pdf"):
        return _fetch_pdf_url(url)

    response = httpx.get(url, headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "application/pdf" in content_type:
        return _fetch_pdf_url(url)
    if "text/html" not in content_type and "text/plain" not in content_type:
        raise ValueError(f"Unsupported content type: {content_type}. Only HTML and PDF URLs are supported.")

    return _parse_html(response.text, url)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")
    if not parsed.netloc:
        raise ValueError("Invalid URL — no domain found")


def _fetch_pdf_url(url: str) -> dict:
    """Download PDF to a temp file and use pdf_service."""
    import tempfile, os
    from services.pdf_service import extract_text_from_pdf

    response = httpx.get(url, headers=_HEADERS, timeout=30, follow_redirects=True)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        result = extract_text_from_pdf(tmp_path)
        result["url"] = url
        result["title"] = url.split("/")[-1] or url
        return result
    finally:
        os.unlink(tmp_path)


def _parse_html(html: str, url: str) -> dict:
    """Parse HTML and extract structured text sections."""
    soup = BeautifulSoup(html, "lxml")

    # Extract page title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        og = soup.find("meta", property="og:title")
        if og:
            title = og.get("content", "").strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    if not title:
        title = urlparse(url).netloc

    # Remove noise tags
    for tag in soup.find_all(_SKIP_TAGS):
        tag.decompose()

    # Try to find the main content container
    content_root = (
        soup.find("article") or
        soup.find("main") or
        soup.find(id=re.compile(r"content|main|article|post|entry", re.I)) or
        soup.find(class_=re.compile(r"content|main|article|post|entry|body", re.I)) or
        soup.body or
        soup
    )

    # Split into logical sections by headings
    sections = _split_into_sections(content_root, title)

    if not sections:
        # Fallback: treat whole page as one section
        raw = _clean_text(content_root.get_text(separator=" "))
        if raw:
            sections = [{"page_number": 1, "text": f"{title}\n\n{raw}"}]

    return {
        "total_pages": len(sections),
        "pages": sections,
        "title": title,
        "url": url,
    }


def _split_into_sections(root: Tag, page_title: str) -> list[dict]:
    """
    Walk the DOM and group text by heading boundaries.
    Each heading starts a new "page" (logical section).
    """
    sections = []
    current_heading = page_title
    current_parts: list[str] = []

    def flush():
        text = _clean_text(" ".join(current_parts))
        if len(text) > 60:  # skip trivially short sections
            sections.append({
                "page_number": len(sections) + 1,
                "text": f"{current_heading}\n\n{text}" if current_heading else text,
            })
        current_parts.clear()

    heading_tags = {"h1", "h2", "h3", "h4"}

    for el in root.descendants:
        if not isinstance(el, Tag):
            continue

        if el.name in heading_tags:
            txt = el.get_text(separator=" ", strip=True)
            if txt:
                flush()
                current_heading = txt

        elif el.name in {"p", "li", "td", "th", "blockquote", "pre", "code"}:
            txt = el.get_text(separator=" ", strip=True)
            if txt and len(txt) > 20:
                current_parts.append(txt)

    flush()  # last section
    return sections


def _clean_text(text: str) -> str:
    """Collapse whitespace, remove zero-width chars, normalize."""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\xa0]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
