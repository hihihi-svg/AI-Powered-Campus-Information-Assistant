"""Inspect raw PDF records: count, fields, and sizes."""
import json, base64

raw_path = "data/raw/campus_documents.jsonl"
records = []
with open(raw_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

pdf_recs = [r for r in records if r.get("content_type", "").upper() == "PDF"]
print(f"Total raw: {len(records)}, PDF: {len(pdf_recs)}")
print(f"PDF fields: {list(pdf_recs[0].keys())}")
print()

for i, r in enumerate(pdf_recs):
    b64 = r.get("raw_bytes_base64", "")
    sz = len(base64.b64decode(b64)) if b64 else 0
    url = r.get("url", "")
    filename = url.split("/")[-1][:60]
    print(f"[{i+1:02d}] {sz:>8} bytes | {r.get('http_status')} | {filename}")
