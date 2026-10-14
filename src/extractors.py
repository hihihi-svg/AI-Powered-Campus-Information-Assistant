"""
extractors.py - Clean, robust data extraction for HTML and PDF campus documents.

Extracts:
- Original / source URL
- Clean page / document title
- Main textual content preserving headings, lists, and sections
- Content type (HTML/PDF)
- Collection timestamp & HTTP status
- Last-modified metadata (from headers or HTML meta tags)
- Strips navigation menus, footers, popups, scripts, and boilerplate
"""

import base64
import io
import logging
import re
from typing import Any, Dict, Optional, Tuple

from bs4 import BeautifulSoup, Comment

from src.ocr import (
    OCR_EMPTY,
    OCR_FAILED,
    OCR_SUCCESS,
    OCR_UNAVAILABLE,
    ocr_pdf_bytes,
)

logger = logging.getLogger("campus_pipeline.extractors")

# Optional PDF parsing libraries
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


class HTMLExtractor:
    """Extracts structured text and metadata from raw HTML pages."""

    BOILERPLATE_TAGS = [
        "script", "style", "noscript", "header", "footer", "nav",
        "svg", "iframe", "button", "form", "aside", "figure"
    ]

    BOILERPLATE_PATTERNS = [
        r"cookie", r"popup", r"modal", r"social", r"share", r"menu-wrapper",
        r"sidebar", r"breadcrumb", r"widget-area", r"comment", r"elementor-location-popup"
    ]

    def __init__(self):
        self.boiler_re = re.compile("|".join(self.BOILERPLATE_PATTERNS), re.IGNORECASE)

    def extract(self, raw_html: str, url: str, headers_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Parse HTML string and return cleaned content and metadata."""
        if not raw_html or not raw_html.strip():
            return {
                "title": None,
                "content": "",
                "last_modified": None,
                "extraction_status": "EMPTY",
                "error_message": "Empty HTML body",
            }

        try:
            soup = BeautifulSoup(raw_html, "html.parser")

            # Remove HTML comments
            for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Extract title before decomposing tags
            title = self._extract_title(soup, url)

            # Extract last-modified from meta tags if available
            last_modified = self._extract_last_modified(soup, headers_meta)

            # Decompose boilerplate tags
            for tag_name in self.BOILERPLATE_TAGS:
                for tag in soup.find_all(tag_name):
                    tag.decompose()

            # Decompose boilerplate classes/ids
            for elem in soup.find_all(True):
                attrs = getattr(elem, "attrs", None)
                if not attrs or not isinstance(attrs, dict):
                    continue
                classes = attrs.get("class", [])
                class_str = " ".join(str(c) for c in classes) if isinstance(classes, list) else str(classes)
                id_str = str(attrs.get("id", ""))
                combined = f"{class_str} {id_str}"
                if self.boiler_re.search(combined):
                    elem.decompose()

            # Preserve headings and extract clean content
            content = self._extract_structured_text(soup)

            status = "SUCCESS" if len(content.strip()) >= 30 else "EMPTY"
            return {
                "title": title,
                "content": content.strip(),
                "last_modified": last_modified,
                "extraction_status": status,
                "error_message": None if status == "SUCCESS" else "Content length below threshold",
            }

        except Exception as e:
            logger.exception(f"Failed to extract HTML content from {url}: {e}")
            return {
                "title": None,
                "content": "",
                "last_modified": None,
                "extraction_status": "FAILED",
                "error_message": str(e),
            }

    def _extract_title(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract high-quality page title from title tag or top heading."""
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            # Clean common brand suffixes like " - RV College of Engineering"
            title = re.sub(r"\s*[\|\-–—]\s*(RV College of Engineering|RVCE).*$", "", title, flags=re.IGNORECASE).strip()

        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(separator=" ", strip=True)

        if not title:
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()

        # Fallback to URL path segment if title still empty
        if not title and url:
            parts = [p for p in url.strip("/").split("/") if p and not p.startswith("http")]
            if parts:
                title = parts[-1].replace("-", " ").replace("_", " ").title()

        return title if title else None

    def _extract_last_modified(self, soup: BeautifulSoup, headers_meta: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract last-modified timestamp from meta tags or HTTP headers."""
        if headers_meta and headers_meta.get("last_modified"):
            return headers_meta["last_modified"]

        for meta_prop in ["article:modified_time", "og:updated_time"]:
            tag = soup.find("meta", property=meta_prop)
            if tag and tag.get("content"):
                return tag["content"].strip()

        for meta_name in ["last-modified", "date", "DC.date.modified"]:
            tag = soup.find("meta", attrs={"name": meta_name})
            if tag and tag.get("content"):
                return tag["content"].strip()

        return None

    def _extract_structured_text(self, soup: BeautifulSoup) -> str:
        """Walk structured elements preserving headings, paragraphs, tables, and lists."""
        blocks = []
        # Walk across the cleaned document tree
        for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table"]):
            # Skip if inside another parsed block to prevent duplication
            if elem.find_parents(["p", "ul", "ol", "table"]):
                continue

            tag = elem.name
            if tag.startswith("h"):
                level = tag[1]
                heading_text = elem.get_text(separator=" ", strip=True)
                if heading_text:
                    prefix = "#" * int(level)
                    blocks.append(f"\n{prefix} {heading_text}\n")
            elif tag == "p":
                p_text = elem.get_text(separator=" ", strip=True)
                if p_text:
                    blocks.append(p_text)
            elif tag in ("ul", "ol"):
                items = []
                for li in elem.find_all("li", recursive=False):
                    li_text = li.get_text(separator=" ", strip=True)
                    if li_text:
                        items.append(f"- {li_text}")
                if items:
                    blocks.append("\n".join(items))
            elif tag == "table":
                rows = []
                for tr in elem.find_all("tr"):
                    cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["th", "td"])]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    blocks.append("\n".join(rows))

        # Deduplicate consecutive identical lines (Elementor desktop/mobile layout artifact)
        clean_blocks = []
        prev = None
        for b in blocks:
            normalized_b = b.strip()
            if normalized_b and normalized_b != prev:
                clean_blocks.append(b)
                prev = normalized_b

        return "\n\n".join(clean_blocks)


class PDFExtractor:
    """Extracts text and metadata from PDF binary content."""

    def extract(self, raw_bytes: bytes, url: str, headers_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract text from PDF using PyMuPDF or pypdf."""
        if not raw_bytes or len(raw_bytes) < 100:
            return {
                "title": None,
                "content": "",
                "last_modified": headers_meta.get("last_modified") if headers_meta else None,
                "extraction_status": "EMPTY",
                "error_message": "PDF payload too small or empty",
            }

        title = None
        last_modified = headers_meta.get("last_modified") if headers_meta else None
        extracted_pages = []

        # Try PyMuPDF first
        if HAS_PYMUPDF:
            try:
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
                meta = doc.metadata or {}
                if meta.get("title") and meta["title"].strip():
                    title = meta["title"].strip()
                if not last_modified and meta.get("modDate"):
                    last_modified = meta["modDate"]

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text and text.strip():
                        extracted_pages.append(text.strip())

                doc.close()
            except Exception as e:
                logger.warning(f"PyMuPDF failed on {url}: {e}. Trying pypdf fallback.")

        # Fallback to pypdf if PyMuPDF failed or produced empty text
        if not extracted_pages and HAS_PYPDF:
            try:
                reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
                meta = reader.metadata
                if not title and meta:
                    if hasattr(meta, "title") and meta.title:
                        title = str(meta.title).strip()
                    elif isinstance(meta, dict) and meta.get("/Title"):
                        title = str(meta["/Title"]).strip()

                for page in reader.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        extracted_pages.append(text.strip())
            except Exception as e:
                logger.warning(f"pypdf extraction failed on {url}: {e}")

        # Native extraction is empty or insufficient: attempt OCR as a fallback for
        # scanned PDFs while keeping the original source URL and metadata intact.
        content = "\n\n--- Page Break ---\n\n".join(extracted_pages).strip()
        ocr_result = None
        if len(content) < 50:
            ocr_result = ocr_pdf_bytes(raw_bytes, url=url)
            ocr_status = ocr_result.get("ocr_status")
            if ocr_status == OCR_SUCCESS:
                ocr_text = (ocr_result.get("ocr_text") or "").strip()
                if ocr_text:
                    extracted_pages = [ocr_text]
                    content = ocr_text
            elif ocr_status in (OCR_EMPTY, OCR_FAILED, OCR_UNAVAILABLE):
                logger.warning(f"OCR fallback did not recover PDF text for {url}: {ocr_result.get('error')}")
                content = content or ""

        # Derive title from filename or first line if title is still missing
        if not title:
            # Check first line of page 1
            if extracted_pages:
                first_lines = extracted_pages[0].splitlines()
                for line in first_lines:
                    line_clean = line.strip()
                    if "page break" in line_clean.lower() or line_clean.startswith("---"):
                        continue
                    if len(line_clean) > 5 and len(line_clean) < 120:
                        title = line_clean
                        break
            if not title and url:
                filename = url.split("/")[-1].replace(".pdf", "").replace("-", " ").replace("_", " ")
                filename = re.sub(r"^\d+\s*", "", filename)
                title = filename.title()

        content = "\n\n--- Page Break ---\n\n".join(extracted_pages).strip()
        status = "SUCCESS" if len(content) >= 50 else "EMPTY"

        # Keep the original metadata and call out OCR-specific failure states when no text is recovered.
        error_message = None if status == "SUCCESS" else "PDF extraction yielded no readable text (scanned or image-only)"
        if not content and ocr_result:
            error_message = ocr_result.get("error") or error_message

        return {
            "title": title,
            "content": content,
            "last_modified": last_modified,
            "extraction_status": status,
            "error_message": error_message,
        }


def extract_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """Unified dispatch extractor function taking raw record and returning processed extraction."""
    url = raw_record.get("url") or raw_record.get("source_url")
    content_type = raw_record.get("content_type", "HTML").upper()
    headers_meta = {
        "last_modified": raw_record.get("last_modified"),
        "http_status": raw_record.get("http_status"),
    }

    if content_type == "PDF":
        extractor = PDFExtractor()
        raw_bytes = raw_record.get("raw_bytes")
        if raw_bytes is None and raw_record.get("raw_bytes_base64"):
            raw_bytes = base64.b64decode(raw_record["raw_bytes_base64"])
        result = extractor.extract(raw_bytes, url, headers_meta)
    else:
        extractor = HTMLExtractor()
        raw_html = raw_record.get("raw_content", "")
        result = extractor.extract(raw_html, url, headers_meta)

    return {
        "source_url": url,
        "title": result.get("title"),
        "content": result.get("content", ""),
        "source_type": content_type,
        "collected_at": raw_record.get("collected_at"),
        "last_modified": result.get("last_modified"),
        "http_status": raw_record.get("http_status"),
        "extraction_status": result.get("extraction_status", "FAILED"),
        "error_message": result.get("error_message"),
    }
