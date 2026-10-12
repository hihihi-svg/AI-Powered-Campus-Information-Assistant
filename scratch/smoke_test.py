"""
smoke_test.py — Retrieval smoke test against the real RVCE dataset.
Runs 10 representative questions, prints top-3 results per query with
actual text excerpts, scores, and source URLs.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import TFIDFRetriever

DATASET = "data/processed/campus_documents.jsonl"

QUESTIONS = [
    # (id, intent, question)
    ("Q-ADM-1", "admissions",               "What is the COMEDK rank cutoff for Computer Science at RVCE?"),
    ("Q-ADM-2", "admissions",               "How do I apply for lateral entry through DCET?"),
    ("Q-FEE-1", "fees_finance",             "What is the tuition fee for B.E. first year at RVCE?"),
    ("Q-FEE-2", "fees_finance",             "What is the late fee penalty for delayed tuition payment?"),
    ("Q-EXA-1", "examinations_regulations", "What is the minimum attendance required for semester exams?"),
    ("Q-EXA-2", "examinations_regulations", "How can I apply for revaluation of my exam paper?"),
    ("Q-DEP-1", "departments_faculty",      "Who is the head of the Computer Science department?"),
    ("Q-PLA-1", "placements_careers",       "What companies recruited from RVCE in the last placement season?"),
    ("Q-FAC-1", "facilities_hostels",       "Is there a hostel facility available for girls at RVCE?"),
    ("Q-GEN-1", "general_college_info",     "What is the address and contact number of RVCE?"),
]

EXCERPT_LEN = 300  # characters to show from chunk text

def run_smoke_test():
    retriever = TFIDFRetriever()
    n_indexed = retriever.load_and_index(DATASET)
    print(f"Indexed {n_indexed} chunks from {DATASET}")
    print(f"{'='*80}\n")

    results_all = {}

    for qid, intent, question in QUESTIONS:
        print(f"[{qid}] ({intent})")
        print(f"  Q: {question}")
        results = retriever.retrieve(question, top_k=3, score_threshold=0.05)

        if not results:
            print("  >> NO RESULTS (below threshold or empty index)")
            results_all[qid] = {
                "question": question,
                "intent": intent,
                "results": [],
                "status": "NO_RESULTS"
            }
        else:
            result_list = []
            for rank, r in enumerate(results, 1):
                excerpt = r["text"][:EXCERPT_LEN].replace("\n", " ").strip()
                print(f"  #{rank}  score={r['relevance_score']:.4f}")
                print(f"       title:   {r['document_title']}")
                print(f"       url:     {r['source_url']}")
                print(f"       topic:   {r['topic']}")
                print(f"       heading: {r['heading']}")
                print(f"       excerpt: {excerpt!r}")
                result_list.append({
                    "rank": rank,
                    "score": r["relevance_score"],
                    "document_title": r["document_title"],
                    "source_url": r["source_url"],
                    "topic": r["topic"],
                    "heading": r["heading"],
                    "excerpt": excerpt,
                })
            results_all[qid] = {
                "question": question,
                "intent": intent,
                "results": result_list,
                "status": "RETRIEVED"
            }
        print()

    # Save raw results for report generation
    os.makedirs("reports", exist_ok=True)
    with open("reports/smoke_test_raw.json", "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved to reports/smoke_test_raw.json")

if __name__ == "__main__":
    run_smoke_test()
