"""Quick inspection of the processed dataset before smoke test."""
import json
from collections import Counter

records = []
with open('data/processed/campus_documents.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Total records: {len(records)}")
print(f"Keys: {list(records[0].keys())}")
print()

# Topic distribution
topics = Counter(r.get('topic') for r in records)
print("Topic distribution:")
for t, c in topics.most_common():
    print(f"  {t!r}: {c}")
print()

# Extraction status
statuses = Counter(r.get('extraction_status') for r in records)
print("Extraction status:")
for s, c in statuses.items():
    print(f"  {s}: {c}")
print()

# Non-empty content stats
non_empty = [r for r in records if r.get('content', '').strip()]
lengths = [len(r['content']) for r in non_empty]
print(f"Non-empty records: {len(non_empty)}")
print(f"Content length: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)//len(lengths)}")
print()

# One title+URL per topic (non-empty only)
print("Sample document per topic (non-empty only):")
seen = set()
for r in records:
    t = r.get('topic')
    if t not in seen and r.get('content', '').strip():
        seen.add(t)
        title = (r.get('title') or '')[:60]
        url = r['source_url']
        print(f"  [{t}] {title!r} -> {url}")
