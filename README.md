# Campus AI Assistant — Knowledge Acquisition & ML Pipeline
### MLOps Engineering & Retrieval-Augmented Generation (RAG) Data Engine

An enterprise-grade, reproducible pipeline that acquires and curates publicly accessible campus information from the official college website ([RV College of Engineering](https://rvce.edu.in/)) for use in a Retrieval-Augmented Generation (RAG) system and Campus Query Intent Classification.

---

## 1. System Architecture

```text
                                  Student Query
                                       │
                                       ▼
                             Campus Intent Classifier
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
            Official Campus RAG                 Community Sources
         (rvce.edu.in Real Data)                        │
                      │                                 │
                      └────────────────┬────────────────┘
                                       ▼
                             Hybrid Retrieval / RAG
                                       │
                                       ▼
                              Campus AI Assistant
                                       │
                                       ▼
                             Source-Backed Response
```

The pipeline implements the **Official Campus RAG Data Engine**, systematically discovering, retrieving, cleaning, validating, and cataloging official campus documents with full provenance and strict reproducibility.

---

## 2. Real-Data Campus Knowledge Acquisition Pipeline

### 2.1 Target Website & Crawl Scope
- **Target URL**: [https://rvce.edu.in/](https://rvce.edu.in/) (R.V. College of Engineering, Bengaluru)
- **Robots Compliance**: Respects `robots.txt` (`https://rvce.edu.in/robots.txt`).
- **Domain Confinement**: Strictly restricted to `rvce.edu.in` and approved subdomains.
- **Priority Domains**:
  - Academics, curriculum, syllabus, and academic calendar
  - Departments (Computer Science, Mechanical, Electronics, MCA, AI/ML, etc.)
  - Admissions, eligibility, and seat allocation
  - Fee payment circulars and scholarship notices
  - Examinations, regulations, and autonomous evaluation
  - Placement records, recruiting companies, and internship statistics
  - Campus facilities (Central Library, sports, laboratories, health centre)
  - Publicly accessible PDF documents (circulars, policies, rank reports)

---

## 3. Pipeline Modules & Architecture

```text
├── config.yaml                   # Central configuration (crawler, storage, cleaning, thresholds)
├── requirements.txt              # Production dependency specifications
├── .gitignore                    # Version control ignore rules
├── src/
│   ├── __init__.py               # Package initializer
│   ├── crawler.py                # Priority BFS crawler, robots.txt, politeness, retry logic
│   ├── extractors.py             # HTML/PDF parsing, boilerplate removal, heading preservation
│   ├── cleaning.py               # Text normalization, deterministic IDs, verifiable topic rules
│   ├── validation.py             # Schema checks, duplicate & near-duplicate detection, QA reporting
│   └── build_dataset.py          # End-to-end pipeline orchestrator & CLI
├── data/
│   ├── raw/
│   │   ├── campus_documents.jsonl    # Raw snapshot with preserved binaries / HTML
│   │   └── collection_manifest.json  # Provenance manifest linking records to source URLs
│   └── processed/
│       └── campus_documents.jsonl    # Cleaned, validated, structured dataset for RAG
└── reports/
    └── data_collection_report.json   # Comprehensive QA metrics and crawl statistics
```

### 3.1 Module Descriptions
1. **[crawler.py](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/crawler.py)**:
   - Manages polite crawling with configurable delay, timeout, and max depth.
   - Enforces `robots.txt` compliance using `RobotsManager`.
   - Uses a priority queue (`heapq`) frontier prioritizing campus-critical endpoints (`department`, `admission`, `fee`, `exam`, `placement`).
   - Dual-mode retrieval for HTML pages and binary PDF documents with exponential backoff retries.
2. **[extractors.py](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/extractors.py)**:
   - Strips boilerplate elements (`<header>`, `<footer>`, `<nav>`, `<aside>`, `<script>`, `<style>`, popups, cookie banners).
   - Preserves semantic document structure: markdown headings (`#`, `##`, `###`), bulleted lists, and tables.
   - Multi-backend PDF parser using PyMuPDF (`fitz`) with `pypdf` fallback.
   - Extracts page titles and HTTP/meta `Last-Modified` timestamps.
3. **[cleaning.py](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/cleaning.py)**:
   - Normalizes unicode characters (NFKC) and cleans formatting artifacts.
   - Eliminates layout-induced duplicate lines (e.g. Elementor responsive mobile/desktop blocks).
   - Generates deterministic, verifiable document IDs (`doc_html_<hash>` / `doc_pdf_<hash>`).
   - Assigns topic labels derived strictly from verifiable URL paths, headings, and metadata (no hallucinated labels; unknown values remain `null`).
4. **[validation.py](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/validation.py)**:
   - Enforces record schema and data types.
   - Exact duplicate detection (URL tracking and SHA-256 content hashing).
   - Near-duplicate detection using Jaccard similarity over word 3-grams with configurable threshold (default `0.85`).
   - Tracks crawl errors, unreadable formats, and extraction statuses (`SUCCESS`, `EMPTY`, `PARTIAL`, `FAILED`).
5. **[build_dataset.py](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/build_dataset.py)**:
   - Coordinates end-to-end execution from crawling to report generation.
   - Maintains `collection_manifest.json` for full dataset provenance.

---

## 4. Dataset Schemas

### 4.1 Processed Dataset (`data/processed/campus_documents.jsonl`)
Each line contains a JSON object conforming strictly to:

| Field | Type | Description |
|---|---|---|
| `document_id` | `str` | Deterministic unique ID (e.g., `doc_html_b9ceb436db517957`) |
| `source_url` | `str` | Validated, canonical URL of the source page/document |
| `title` | `str \| null` | Extracted page title or document title |
| `content` | `str` | Cleaned textual content preserving headings and sections |
| `source_type` | `str` | Document format: `"HTML"` or `"PDF"` |
| `collected_at` | `str` | ISO 8601 UTC timestamp of acquisition |
| `last_modified`| `str \| null` | Server/metadata last modification timestamp |
| `topic` | `str \| null` | Verifiable topic classification (or `null` if unclassified) |
| `extraction_status` | `str` | `"SUCCESS"`, `"EMPTY"`, `"PARTIAL"`, or `"FAILED"` |

### 4.2 Raw Dataset & Provenance Manifest
- **`data/raw/campus_documents.jsonl`**: Complete raw capture including full HTML text and base64-encoded PDF binaries.
- **`data/raw/collection_manifest.json`**: Cryptographic provenance manifest linking each `document_id` to its source URL, extraction status, content length, and crawl configuration.

---

## 5. Execution & Reproducibility

### 5.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/hihihi-svg/AI-Powered-Campus-Information-Assistant.git
cd AI-Powered-Campus-Information-Assistant

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Dataset Versioning with DVC
The repository uses DVC to version the canonical dataset files without altering their contents. No remote storage is configured in this workspace.

```bash
# Install DVC if it is not available
python -m pip install dvc

# Initialize DVC once per repository
dvc init

# Track the canonical raw and processed dataset files
# Leave the files on disk; DVC creates the .dvc metadata and gitignore entries.
dvc add data/raw/campus_documents.jsonl data/raw/collection_manifest.json data/processed/campus_documents.jsonl

# Inspect DVC state locally
dvc status

# Inspect Git state after tracking metadata is created
git status --short --branch
```

> The generated local artifacts such as `mlruns/` and synthetic degradation outputs are intentionally kept out of source data tracking. No remote backup is configured or implied.

### 5.3 Running the Pipeline
Run the complete pipeline using `config.yaml`:
```bash
python -m src.build_dataset
```

Override crawl limits from the command line:
```bash
# Crawl 30 pages
python -m src.build_dataset --max-pages 30

# Specify custom configuration
python -m src.build_dataset --config config.yaml --max-pages 60
```

### 5.4 Local Demo UI
The repository includes a lightweight browser UI for the existing FastAPI backend. The app serves the frontend at the root URL and keeps the API endpoints working:

```bash
cd "C:\Users\lalan\Desktop\AI-Powered-Campus-Information-Assistant"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Then open the app in a browser:

```text
http://127.0.0.1:8000/
```

Available endpoints include:
- `GET /health`
- `POST /ask`
- `GET /docs`
- `GET /`

The root page is a plain HTML/CSS/JavaScript interface that submits questions to the backend `/ask` endpoint and displays the response, citations, and abstention state without duplicating the retrieval logic.

---

## 6. Actual Live Collection Statistics & Findings

Results from running the pipeline against [https://rvce.edu.in/](https://rvce.edu.in/):

| Metric | Result | Notes |
|---|---|---|
| **Total Documents Collected** | **60** | 45 HTML pages + 15 PDF documents |
| **HTML Pages Retrieved** | **45** | Departments, admissions, circulars, fees, academics |
| **PDF Documents Retrieved** | **15** | Official circulars, fee notifications, rankings |
| **Successful Extractions** | **45** | Fully parsed structured content |
| **Empty Extractions** | **15** | Scanned circular notices without embedded OCR layer |
| **Failed Extractions** | **0** | Zero unhandled parser exceptions |
| **Crawl Errors** | **1** | Tracked and logged in QA report |
| **Exact Content Duplicates** | **3** | Detected URL aliases (e.g. `/academic_student_services/`) |
| **Near-Duplicates (Jaccard $\ge 0.85$)** | **17** | Identified shared department academic planning templates |
| **Mean Content Length** | **3,144 characters** | Preserving headings and bullet points |
| **Mean Word Count** | **475 words** | Clean body text without navigational noise |

### Topic Distribution
- `departments`: 31 documents (CSE, ME, ECE, MCA, AI/ML, Chemistry, Chemical Eng, etc.)
- `admissions`: 10 documents (Circulars, eligibility, management quota details)
- `fees_finance`: 9 documents (Higher semester circulars, payment procedures)
- `examinations_regulations`: 6 documents (Circulars, autonomous exam guidelines)
- `academics`: 2 documents (Academic student services, curriculum)
- `placements_careers`: 1 document (Placement & training overview)
- `unclassified`: 1 document (Homepage hub)

---

## 7. Quality Assurance Report (`reports/data_collection_report.json`)

The automated QA engine generates a complete inspection report with:
- Collection summary and schema validation counts
- Breakdown by document type and verifiable topic
- Exact duplicate detection with source URLs
- Near-duplicate pairs with Jaccard similarity metrics
- Content length distribution (min, max, mean, median)
- Crawl error breakdown and status codes

---

## 8. Downstream Intent Classification Roadmap

The acquired campus documents serve as the authoritative knowledge base for RAG grounding. Student queries are routed via the Campus Query Intent Classifier:

| Intent | Description |
|---|---|
| `ACADEMICS` | Courses, examinations, attendance, timetable, electives, regulations |
| `ADMINISTRATION` | Certificates, applications, documents, administrative procedures |
| `HOSTEL` | Accommodation, rules, timings, mess facilities |
| `FACILITIES` | Library, laboratories, sports, campus infrastructure |
| `FEES_FINANCE` | Tuition fees, scholarships, payment circulars |
| `EVENTS_CLUBS` | Student clubs, technical chapters, cultural teams |
| `PLACEMENTS` | Campus recruitment, internship statistics, eligibility |
| `STUDENT_EXPERIENCE` | Campus life, food, peer guidance |
| `OUT_OF_SCOPE` | Queries unrelated to campus services |

## 9. Local Docker and CI

### 9.1 Build the image

```bash
docker build -t rvce-campus-assistant .
```

### 9.2 Run the API container

```bash
docker run --rm -p 8000:8000 rvce-campus-assistant
```

Then open:
- `/health` for health status
- `/docs` for the FastAPI Swagger UI

Example:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/docs
```

### 9.3 GitHub Actions CI

The repository includes a minimal GitHub Actions workflow at [./.github/workflows/ci.yml](.github/workflows/ci.yml). It installs dependencies and runs `pytest -q` for every push and pull request without requiring external services or API keys.

---

## 10. Multi-Strategy Retrieval, MLflow Model Tracking & Evaluation

### 10.1 Retrieval Configurations
The assistant supports four retrieval configurations through `src/retrieval.py`:
1. **TF-IDF Baseline (`TFIDFRetriever`)**: Lexical n-gram vectorizer with sublinear term-frequency, heading-aware chunking, boilerplate suppression, and stable deduplication.
2. **Dense Semantic Retrieval (`SemanticRetriever`)**: Uses `sentence-transformers/all-MiniLM-L6-v2` on CPU with in-memory corpus embedding caching and normalized cosine similarity.
3. **Hybrid Retrieval (`HybridRetriever`)**: Combines lexical and semantic candidates without chunk duplication, supporting both normalized linear score combination and Reciprocal Rank Fusion (RRF).
4. **Cross-Encoder Reranking (`CrossEncoderReranker`)**: Evaluates candidate passages with `cross-encoder/ms-marco-MiniLM-L6-v2`, strictly preserving source metadata, with safe fallback when model weights are not cached locally.

### 10.2 MLflow Tracking & Model Artifact Logging
- **Experiment Tracking**: Records runs under `rvce-campus-retrieval-benchmark` and `rvce-campus-retrieval-benchmark-labelled`.
- **Model Logging**: Logs loadable SentenceTransformer models using native `mlflow.sentence_transformers.log_model` and verifies reload through automated inference smoke tests.
- **Model Registry**: Models can be registered in the local MLflow Model Registry (`rvce-sentence-transformer`).
- **Parameter & Metric Governance**: Logs retriever types, fusion weights, threshold parameters, execution devices, file checksums, duration, and evaluation metrics.

### 10.3 Evaluation Methodology & Empirical Results

#### Benchmark 1: Unlabelled 40-Query Benchmark (`campus_queries.csv`)
*Evaluates score threshold hit proxy (`retrieval_coverage`). It does not contain gold relevance labels.*

| Configuration | Retrieval Coverage | Avg Best Score | Avg Results | Duration |
|---|---|---|---|---|
| `tfidf-baseline` | **1.0000** | 0.1474 | 2.6250 | **0.24s** |
| `semantic-minilm` | **1.0000** | **0.4739** | **3.0000** | 22.66s |
| `hybrid-minilm` | **1.0000** | 0.2859 | **3.0000** | 11.75s |
| `hybrid-minilm-cross-encoder` | **1.0000** | 0.2859 | **3.0000** | 12.14s |

#### Benchmark 2: Curated 8-Query Labelled Relevance Benchmark (`labelled_relevance_benchmark.json`)
*Measures standard ranking metrics against human-curated ground-truth documents across campus domains.*

| Configuration | Recall@3 | Precision@3 | MRR | nDCG@3 |
|---|---|---|---|---|
| **`tfidf-baseline`** | **0.6250** | **0.2083** | **0.2917** | **0.4539** |
| **`semantic-minilm`** | 0.5000 | 0.1667 | 0.2500 | 0.3914 |
| **`hybrid-minilm`** | 0.5000 | 0.1667 | 0.2500 | 0.3914 |

**Empirical Takeaway**: In this campus knowledge domain, the TF-IDF baseline achieves higher Recall@3 and nDCG@3 than out-of-the-box dense semantic retrieval due to exact college acronyms and administrative terminology ("COMEDK", "VTU", "B.E. first year", "SSP portal"). Consequently, the production API conservatively retains `TFIDFRetriever` as the default.

### 10.4 Reproduction Commands

#### Run the entire test suite (44 tests):
```powershell
.venv-hf-fix\Scripts\python.exe -m pytest tests/
```

#### Run and compare all 4 retrieval configurations in MLflow:
```powershell
.venv-hf-fix\Scripts\python.exe -m src.evaluation --run-all-configs
```

#### Run labelled ground-truth evaluation:
```powershell
.venv-hf-fix\Scripts\python.exe -m src.evaluation --run-labelled --retriever tfidf
.venv-hf-fix\Scripts\python.exe -m src.evaluation --run-labelled --retriever semantic
.venv-hf-fix\Scripts\python.exe -m src.evaluation --run-labelled --retriever hybrid
```

#### Launch local MLflow UI:
```powershell
.venv-hf-fix\Scripts\python.exe -m mlflow ui --port 5000
```
Then navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to view runs, parameters, metrics, and logged model artifacts.


