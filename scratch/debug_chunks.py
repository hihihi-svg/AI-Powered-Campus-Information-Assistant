"""Inspect actual chunk content to find root causes of bad results."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval import TFIDFRetriever, chunk_document

# Load raw records
records = []
with open('data/processed/campus_documents.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# Find the "B.E first year Academic circulars" record and the "Fee payment circulars" record
# that are causing bad top results for Q-FEE-1 and Q-FEE-2
problem_urls = [
    "https://rvce.edu.in/b-e-first-year-academic-circulars/",
    "https://rvce.edu.in/academics_and_examinations/fee_payment_circulars/",
    "https://rvce.edu.in/academic_student_services/",
]

for r in records:
    if r['source_url'] in problem_urls:
        print(f"\n{'='*60}")
        print(f"URL: {r['source_url']}")
        print(f"Title: {r['title']}")
        print(f"Topic: {r['topic']}")
        print(f"Status: {r['extraction_status']}")
        print(f"Content (first 800 chars):")
        print(repr(r['content'][:800]))
        
        print("\n  -- CHUNKS --")
        chunks = chunk_document(r)
        for c in chunks:
            print(f"  [{c.chunk_id}] heading={c.heading!r} len={c.char_length}")
            print(f"    text={c.text[:150]!r}")
