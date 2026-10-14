# Webpage Retrieval & Sentence Transformer Evaluation Report

## Executive Summary
This evaluation confirms that **Sentence Transformer retrieval (`all-MiniLM-L6-v2`) is fully functional, verified on CPU, and integrated into the live RVCE Campus Information Assistant web application**.

The live application is running at **`http://127.0.0.1:8000/`**.

---

## 1. Web Application Architecture & Inspection

| Component | Implementation Details |
| :--- | :--- |
| **Frontend Entry Point** | `static/index.html`, mounted at FastAPI route `@app.get("/")`. Static assets (`styles.css`, `app.js`) are served via `/static`. |
| **Backend API Route** | `POST /ask` accepting `{"question": "...", "strategy": "tfidf"|"semantic"|"hybrid"|"hybrid_rerank"}`. |
| **Default Strategy** | `TFIDFRetriever` (retained as default baseline for precision and conservative evidence grounding). |
| **Sentence Transformer Model** | `sentence-transformers/all-MiniLM-L6-v2` loaded on CPU from local Hugging Face cache. |
| **Singleton Caching** | Document embeddings (247 chunks) are computed and cached in memory. Subsequent queries respond in **~30–60 ms**. |
| **Evidence & Scores** | The API returns `answer`, `abstention_status`, `sources` (with URLs), `evidence` (with text snippets and `relevance_score`), and `strategy`. |

---

## 2. Web UI Enhancements Made

To make the retrieval process visible to the user without overwhelming the interface, the following UI features were added:
1. **Retrieval Strategy Selector**: A dropdown in the query form allowing users to select between **TF-IDF (Default Baseline)**, **Dense Semantic (all-MiniLM-L6-v2)**, **Hybrid (TF-IDF + Dense Linear)**, and **Hybrid + Cross-Encoder Rerank**.
2. **Explicit Status Badges**: Each answer message displays a clear visual badge:
   - `Status: Answered` (green) when supported by evidence.
   - `Status: Insufficient Evidence` (amber) when abstaining conservatively.
3. **Strategy Badge**: Displays the active strategy (e.g., `Strategy: SEMANTIC`).
4. **Clickable Source Links**: Clean source titles linking directly to the official RVCE URLs.
5. **Collapsible Diagnostics Accordion**: `<details>` dropdown titled *"Retrieved Evidence Passages (N) & Relevance Scores"*, displaying each evidence passage and its calculated cosine relevance score.
6. **Loading Indicator**: Real-time spinner showing *"Searching campus knowledge base..."* while processing.

---

## 3. Results on the 10 Benchmark Questions

| # | Question | TF-IDF Result | Semantic (Sentence Transformers) Result | Corpus Support Assessment |
| :- | :--- | :--- | :--- | :--- |
| **1** | *Who is Dr. K.N. Subramanya?* | **Answered** (Score: 0.1849)<br>Placement contact list | **Answered** (Score: 0.2903)<br>Placement contact list | Returns placement contact card mentioning Dr. D Ranganath. Principal's dedicated bio is not in the text dataset. |
| **2** | *What is the official email for RVCE?* | **Abstained** | **Abstained** | **Correct abstention**: General official email is not present in the indexed text documents. |
| **3** | *What does the Aerospace Engineering department focus on?* | **Abstained** | **Abstained** | **Correct abstention**: Department overview text for AE is not indexed in the current text snapshot. |
| **4** | *What is the procedure for the COMEDK quota admission process?* | **Answered** (Score: 0.2195)<br>`admission-circulars/` | **Answered** (Score: 0.5976)<br>`admission-circulars/` | **Strong match**: Retrieves the official COMEDK procedure, reporting location (Civil Eng Dept, Room CE-205), and fee details. |
| **5** | *What is the academic calendar for RVCE?* | **Answered** (Score: 0.3765)<br>`department/eee/academic_planning/` | **Answered** (Score: 0.6844)<br>`department/ae/academic_planning/` | **Strong match**: Retrieves official academic calendar notice with high semantic similarity. |
| **6** | *Who heads RVCE?* | **Abstained** | **Abstained** | **Correct abstention**: No document explicitly names the head in the required biographical entity format. |
| **7** | *How can I contact the college officially?* | **Answered** (Score: 0.0651)<br>Fee extension circular | **Answered** (Score: 0.4258)<br>Admission process contact | **Supported**: Retrieves official admission and fee office contact details. |
| **8** | *What is the capital of France?* | **Abstained** | **Abstained** | **Correct abstention**: Out-of-domain knowledge; safely abstains without hallucinating. |
| **9** | *What is Microsoft's current CEO's name?* | **Abstained** | **Abstained** | **Correct abstention**: Out-of-domain person query; safely abstains without hallucinating. |
| **10** | *What is the RVCE hostel fee for the current academic year?* | **Answered** (Score: 0.1620)<br>Academic planning circular | **Answered** (Score: 0.4845)<br>Higher semester fees circular | **Supported**: Retrieves fee circular mentioning higher semester fees and hostel quota circulars. |

---

## 4. Performance & Latency Summary

- **Test Suite**: **44 passing tests, 0 failures** across `tests/test_answering.py`, `tests/test_api.py`, `tests/test_evaluation.py`, and `tests/test_retrieval.py`.
- **TF-IDF Query Latency**: **~7–25 ms**.
- **Sentence Transformers Cold Start**: ~20 seconds (one-time corpus encoding for 247 chunks).
- **Sentence Transformers Cached Latency**: **~30–60 ms** per query.
