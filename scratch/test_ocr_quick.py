"""Quick test: OCR on first PDF from raw dataset."""
import json, base64, sys, os
sys.path.insert(0, ".")

from src.ocr import ocr_pdf_bytes, is_rapidocr_available

print(f"RapidOCR available: {is_rapidocr_available()}")

# Load first PDF
records = []
with open("data/raw/campus_documents.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

pdf_recs = [r for r in records if r.get("content_type", "").upper() == "PDF"]

# Test on first 2 PDFs, limiting to 2 pages each
for i, r in enumerate(pdf_recs[:2]):
    b64 = r.get("raw_bytes_base64", "")
    raw_bytes = base64.b64decode(b64) if b64 else b""
    url = r.get("url", "")
    filename = url.split("/")[-1][:50]

    print(f"\n--- PDF {i+1}: {filename} ({len(raw_bytes)} bytes) ---")
    result = ocr_pdf_bytes(raw_bytes, url=url, dpi=150, max_pages=2)
    print(f"  ocr_status      : {result['ocr_status']}")
    print(f"  page_count      : {result['page_count']}")
    print(f"  pages_with_text : {result['pages_with_text']}")
    print(f"  error           : {result['error']}")
    text = result.get("ocr_text", "")
    print(f"  text_chars      : {len(text)}")
    if text:
        print(f"  text_excerpt    : {text[:400]!r}")
