"""
ocr.py — OCR module for extracting text from scanned PDF pages.

Strategy:
  1. PyMuPDF rasterizes each PDF page to a numpy pixel array (no temp files).
  2. RapidOCR (ONNX-runtime, CPU-only, no GPU required) runs inference on each page image.
  3. Falls back gracefully: if RapidOCR is unavailable, status is marked OCR_UNAVAILABLE.
  4. Page text is joined with page-break markers matching the existing PDFExtractor convention.
  5. Short / empty OCR output is marked OCR_EMPTY — never silently treated as success.
  6. Original URL, title, document_id, and all metadata are NEVER modified here.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("campus_pipeline.ocr")

# ──────────────────────────────────────────────
# Dependency availability flags
# ──────────────────────────────────────────────
try:
    import fitz  # PyMuPDF — for rasterising PDF pages
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    logger.warning("PyMuPDF (fitz) not available; OCR pipeline cannot rasterize PDFs.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

_rapidocr_engine = None   # lazy singleton
_RAPIDOCR_AVAILABLE: Optional[bool] = None  # None = untested


def _get_rapidocr():
    """Return a RapidOCR engine singleton, or None if unavailable."""
    global _rapidocr_engine, _RAPIDOCR_AVAILABLE

    if _RAPIDOCR_AVAILABLE is True:
        return _rapidocr_engine

    if _RAPIDOCR_AVAILABLE is False:
        return None

    # First call — try to import and instantiate
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        # Smoke-test: run on a tiny blank image to confirm ONNX runtime works
        import numpy as np
        blank = np.zeros((64, 256, 3), dtype=np.uint8)
        blank[:] = 255  # white
        result, _ = engine(blank)
        _rapidocr_engine = engine
        _RAPIDOCR_AVAILABLE = True
        logger.info("RapidOCR engine initialised and smoke-tested successfully.")
        return _rapidocr_engine
    except ImportError:
        logger.warning("rapidocr-onnxruntime not installed. OCR will be unavailable.")
        _RAPIDOCR_AVAILABLE = False
        return None
    except Exception as e:
        logger.warning(f"RapidOCR initialisation failed: {e}. OCR will be unavailable.")
        _RAPIDOCR_AVAILABLE = False
        return None


# ──────────────────────────────────────────────
# Public constants
# ──────────────────────────────────────────────
OCR_SUCCESS     = "OCR_SUCCESS"
OCR_EMPTY       = "OCR_EMPTY"        # ran OCR but got no usable text
OCR_FAILED      = "OCR_FAILED"       # runtime error during OCR
OCR_UNAVAILABLE = "OCR_UNAVAILABLE"  # library not installed
OCR_SKIPPED     = "OCR_SKIPPED"      # native text already present; OCR not needed

# Minimum character count to consider OCR output usable
MIN_OCR_CHARS = 50


def _rasterize_pdf_pages(raw_bytes: bytes, dpi: int = 200) -> List[Any]:
    """
    Convert each PDF page to a numpy RGB array using PyMuPDF.

    Args:
        raw_bytes: Raw PDF binary content.
        dpi: Rendering resolution (200 is a good balance for text readability vs speed).

    Returns:
        List of numpy arrays (H, W, 3), one per page.
    """
    if not HAS_FITZ or not HAS_NUMPY:
        return []

    images = []
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        zoom = dpi / 72.0  # 72 DPI is PyMuPDF default
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            # Convert pixmap to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            images.append(img_array)

        doc.close()
    except Exception as e:
        logger.warning(f"PDF rasterization failed: {e}")

    return images


def _ocr_single_image(engine: Any, img_array: Any) -> str:
    """
    Run RapidOCR on one page image and return the concatenated text.

    Returns empty string on failure.
    """
    try:
        result, _ = engine(img_array)
        if not result:
            return ""
        # result is a list of [bbox, text, confidence] triples
        lines = [item[1] for item in result if item and len(item) >= 2 and item[1]]
        return " ".join(lines).strip()
    except Exception as e:
        logger.warning(f"RapidOCR inference error on page: {e}")
        return ""


def _clean_ocr_text(raw: str) -> str:
    """
    Light post-processing of raw OCR output:
    - Collapse multiple whitespace runs.
    - Remove lines that are just noise (single chars, page numbers only).
    - Do NOT modify or invent textual content.
    """
    if not raw:
        return ""

    # Normalise Unicode
    import unicodedata
    raw = unicodedata.normalize("NFKC", raw)

    # Collapse whitespace within lines
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        # Drop very short noise lines (single characters, lone digits, isolated punctuation)
        if len(line) <= 1:
            continue
        if re.fullmatch(r"[\d\s\-–—\.,:;]+", line) and len(line) < 5:
            continue
        lines.append(line)

    return "\n".join(lines)


def ocr_pdf_bytes(
    raw_bytes: bytes,
    url: str = "",
    dpi: int = 200,
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run OCR on a scanned PDF binary.

    This should only be called when native PDF text extraction has already
    returned empty content.

    Args:
        raw_bytes: Raw PDF binary.
        url: Source URL for logging purposes only.
        dpi: Rasterization resolution (default 200).
        max_pages: If set, limits pages processed (useful for large PDFs during testing).

    Returns:
        Dict with keys:
            ocr_status  : one of OCR_SUCCESS, OCR_EMPTY, OCR_FAILED, OCR_UNAVAILABLE
            ocr_text    : Extracted text (empty string if failed)
            page_count  : Number of pages rasterized
            pages_with_text: Number of pages that returned non-empty OCR output
            error       : Error message if failed, else None
    """
    if not raw_bytes or len(raw_bytes) < 100:
        return {
            "ocr_status": OCR_FAILED,
            "ocr_text": "",
            "page_count": 0,
            "pages_with_text": 0,
            "error": "PDF payload too small or empty",
        }

    engine = _get_rapidocr()
    if engine is None:
        return {
            "ocr_status": OCR_UNAVAILABLE,
            "ocr_text": "",
            "page_count": 0,
            "pages_with_text": 0,
            "error": "rapidocr-onnxruntime not available",
        }

    # Rasterize pages
    page_images = _rasterize_pdf_pages(raw_bytes, dpi=dpi)
    if not page_images:
        return {
            "ocr_status": OCR_FAILED,
            "ocr_text": "",
            "page_count": 0,
            "pages_with_text": 0,
            "error": "Could not rasterize PDF pages (fitz/numpy unavailable or corrupt PDF)",
        }

    # Optionally limit pages (for testing / large circulars)
    if max_pages is not None:
        page_images = page_images[:max_pages]

    page_texts = []
    pages_with_text = 0

    for i, img in enumerate(page_images):
        raw_text = _ocr_single_image(engine, img)
        cleaned = _clean_ocr_text(raw_text)
        if cleaned.strip():
            page_texts.append(cleaned)
            pages_with_text += 1
            logger.debug(f"  Page {i+1}: {len(cleaned)} chars OCR'd")
        else:
            page_texts.append("")  # preserve page boundary
            logger.debug(f"  Page {i+1}: no text extracted")

    combined = "\n\n--- Page Break ---\n\n".join(
        t for t in page_texts if t.strip()
    )
    combined = combined.strip()

    if len(combined) < MIN_OCR_CHARS:
        return {
            "ocr_status": OCR_EMPTY,
            "ocr_text": "",
            "page_count": len(page_images),
            "pages_with_text": pages_with_text,
            "error": f"OCR output too short ({len(combined)} chars < {MIN_OCR_CHARS} threshold)",
        }

    logger.info(
        f"OCR success: {url!r} — {len(page_images)} pages, "
        f"{pages_with_text} with text, {len(combined)} chars total"
    )
    return {
        "ocr_status": OCR_SUCCESS,
        "ocr_text": combined,
        "page_count": len(page_images),
        "pages_with_text": pages_with_text,
        "error": None,
    }


def is_rapidocr_available() -> bool:
    """Return True if RapidOCR is installed and passed its smoke test."""
    return _get_rapidocr() is not None
