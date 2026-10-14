# RVCE Campus Information Assistant — Retrieval Configuration & MLflow Tracking Report

**Date**: 2026-10-14  
**Author**: Senior MLOps Engineer  
**Tracking URI**: Local file-backed store (`./mlruns`)  
**MLflow Experiment**: `rvce-campus-retrieval-benchmark` & `rvce-campus-retrieval-benchmark-labelled`  
**Target Knowledge Corpus**: 60 documents (45 HTML, 15 PDF snapshots), 247 indexed passage chunks  

---

## 1. Executive Summary

This report documents the design, implementation, and empirical evaluation of multi-strategy retrieval and MLflow model tracking for the RVCE Campus Information Assistant. We implemented:
1. **Configuration A (TF-IDF Baseline)**: Lexical n-gram matching with sublinear term-frequency and cosine similarity.
2. **Configuration B (Semantic Retrieval)**: Dense embedding retrieval using `sentence-transformers/all-MiniLM-L6-v2` with in-memory corpus embedding caching and normalized cosine similarity.
3. **Configuration C (Hybrid Retrieval)**: Lexical + semantic candidate fusion supporting both linear score combination and Reciprocal Rank Fusion (RRF), eliminating duplicate document chunks.
4. **Configuration D (Hybrid with Cross-Encoder Reranking)**: Candidate passage reranking using `cross-encoder/ms-marco-MiniLM-L6-v2`, with safe fallback when model weights are not cached locally.
5. **MLflow Model Artifact Logging & Registry**: Native SentenceTransformer model logging, model reload verification on CPU, and Model Registry support.
6. **Rigorous Evaluation Methodology**: Clear distinction between the 40-query unlabelled intent benchmark (coverage proxy) and a curated 8-query ground-truth relevance benchmark (Recall@k, Precision@k, MRR, nDCG@k).

---

## 2. Dependency Audit & Environment Resolution

### Historical Dependency Conflict
Previously, the global environment suffered from:
- `huggingface-hub==1.4.1` installed alongside `transformers==4.57.1`, which caused a hard import failure (`huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module, but found huggingface-hub==1.4.1`).
- Consequently, `sentence-transformers` could not be loaded in the base Python environment.

### Solution & Pinned Environment
To avoid modifying unrelated system packages and risking breakages in existing projects, we utilized the project-specific virtual environment `.venv-hf-fix`:
- Python version: `3.11.9` (64-bit on Windows)
- PyTorch: `2.14.1+cpu`
- Transformers: `4.57.6`
- Hugging Face Hub: `0.36.2` (resolves the `<1.0` constraint)
- Sentence Transformers: `5.7.0`
- Tokenizers: `0.22.2`
- MLflow: `3.17.1` (with native `mlflow.sentence_transformers` flavor)
- FastAPI: `0.143.0`
- Scikit-learn: `1.9.1`
- Pytest: `9.1.1`

All direct dependencies were declared in [requirements.txt](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/requirements.txt).

### CPU Inference Smoke Test
- `sentence-transformers/all-MiniLM-L6-v2`: **Verified on CPU using cached local weights** (`C:\Users\lalan\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2`).
  - Output embedding shape: `(2, 384)`
  - Finite numerical values: Verified (all values finite)
  - Normalization: Verified (L2 norm = 1.0000)
- `cross-encoder/ms-marco-MiniLM-L6-v2`: Model weights were **not present in the local cache** and remote download was not attempted. The reranker graceful fallback was smoke-tested and verified: when weights are absent, it logs a clear warning, retains original hybrid candidate scores and metadata, and completes retrieval without crashing.

---

## 3. Retrieval Architecture & Configuration Details

### Configuration A: TF-IDF Baseline (`TFIDFRetriever`)
- Vectorizer: `TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)`
- Chunking: Section- and heading-aware chunking with site-wide boilerplate filtering (filtering footer blocks, navigation menus, and widget lists).
- Deduplication: Normalized passage deduplication key `(document_id, heading, normalized_text)`.

### Configuration B: Dense Semantic Retrieval (`SemanticRetriever`)
- Model: `sentence-transformers/all-MiniLM-L6-v2` on CPU.
- Optimization: Corpus embeddings for all 247 chunks are encoded once and cached in memory (`self.document_embeddings`).
- Performance: First run encodes corpus in ~22 seconds; all subsequent queries execute in **~30 milliseconds**.

### Configuration C: Hybrid Retrieval (`HybridRetriever`)
- Candidates: Retrieves top candidates from both lexical and dense engines.
- Deduplication: Merges candidates without duplicate passages using stable deduplication signatures.
- Configurable Fusion Methods:
  1. `linear`: Normalized weighted combination:
     $$\text{Score} = w_{\text{tfidf}} \cdot S_{\text{lexical}} + w_{\text{semantic}} \cdot S_{\text{semantic}}$$
     (Default: $w_{\text{tfidf}} = 0.5, w_{\text{semantic}} = 0.5$).
  2. `rrf`: Reciprocal Rank Fusion:
     $$\text{Score} = \frac{w_{\text{tfidf}}}{k + \text{rank}_{\text{tfidf}}} + \frac{w_{\text{semantic}}}{k + \text{rank}_{\text{semantic}}}$$
     (Default: $k = 60$).

### Configuration D: Cross-Encoder Reranking (`CrossEncoderReranker`)
- Reranks top 15-20 hybrid candidates against query text.
- Metadata preservation: Strictly preserves `chunk_id`, `document_id`, `source_url`, `document_title`, `heading`, and `topic`.
- Controlled Fallback: If model is unavailable, gracefully returns hybrid candidates.

---

## 4. Empirical Evaluation Results

### 4.1 Benchmark 1: 40-Query Campus Query Benchmark (`campus_queries.csv`)
*Note: This dataset contains query text and intent labels only; it does not contain gold relevance labels. The `retrieval_coverage` (relevance_rate) measures whether at least one retrieved chunk scored above threshold ($s \ge 0.05$).*

| Configuration | Retrieval Coverage | Average Best Score | Average Retrieved Results | Evaluation Duration |
|---|---|---|---|---|
| **TF-IDF Baseline** | **1.0000 (40/40)** | 0.1474 | 2.6250 | **0.2426s** |
| **Semantic (MiniLM)** | **1.0000 (40/40)** | **0.4739** | **3.0000** | 22.6609s (with MLflow model log) |
| **Hybrid (Linear)** | **1.0000 (40/40)** | 0.2859 | **3.0000** | 11.7526s |
| **Hybrid (Rerank Fallback)** | **1.0000 (40/40)** | 0.2859 | **3.0000** | 12.1399s |

### 4.2 Benchmark 2: Curated Labelled Relevance Benchmark (`labelled_relevance_benchmark.json`)
*Curated 8-query ground-truth benchmark mapped to known target document IDs and source URLs across Admissions, Fees, Exams, Placements, Faculty, and Student Services.*

| Configuration | Recall@3 | Precision@3 | MRR (Mean Reciprocal Rank) | nDCG@3 |
|---|---|---|---|---|
| **TF-IDF Baseline** | **0.6250** | **0.2083** | **0.2917** | **0.4539** |
| **Semantic (MiniLM)** | 0.5000 | 0.1667 | 0.2500 | 0.3914 |
| **Hybrid (Linear)** | 0.5000 | 0.1667 | 0.2500 | 0.3914 |

### Critical MLOps Finding:
In this campus domain, **the TF-IDF baseline outperforms out-of-the-box dense semantic retrieval on labelled ranking metrics** (Recall@3 of 0.6250 vs 0.5000, nDCG@3 of 0.4539 vs 0.3914).
- **Reason**: Official campus queries rely heavily on exact acronyms and specialized tokens ("COMEDK", "VTU", "B.E. first year", "SSP portal", "fast-track examinations"). General-purpose MiniLM dense embeddings spread attention over broader semantic contexts, occasionally retrieving generic pages ahead of exact circulars.
- **Architectural Decision**: The production API continues to use `TFIDFRetriever` as its verified default backend, allowing clients to opt into semantic or hybrid retrieval via optional parameters.

---

## 5. MLflow Tracking, Model Artifact Logging, and Model Registry

### Experiment Organization
- Tracking Directory: `./mlruns`
- Experiments:
  - `rvce-campus-retrieval-benchmark`: Houses benchmark runs for all 4 retrieval configurations.
  - `rvce-campus-retrieval-benchmark-labelled`: Houses runs evaluating ground-truth ranking metrics.
- Runs logged:
  - `tfidf-baseline`
  - `semantic-minilm`
  - `hybrid-minilm`
  - `hybrid-minilm-cross-encoder`

### Logged Parameters
`retriever_type`, `sentence_transformer_model`, `cross_encoder_model`, `embedding_dimension`, `similarity_metric`, `fusion_method`, `semantic_weight`, `tfidf_weight`, `top_k`, `score_threshold`, `rerank`, `dataset_path`, `benchmark_path`, `benchmark_checksum`, `benchmark_has_relevance_labels`, `python_version`, `mlflow_version`, `execution_device`, `git_commit`.

### Model Artifact Logging & Reload Verification
For semantic and hybrid runs, the actual loadable Sentence Transformer model was logged using `mlflow.sentence_transformers`:
- Artifact path: `sentence_transformer_model`
- Model URI format: `models:/m-...` or file artifact URI
- **Reload Smoke Test**: Re-loaded the model from MLflow using `mlflow.sentence_transformers.load_model(model_uri)` and executed CPU inference, confirming embedding shape `(1, 384)` and norm `1.0000`.
- Tag: `model_reload_verified=true`.

### Model Registry Status
In MLflow 3.17.1 with `MLFLOW_ALLOW_FILE_STORE=true`, local file-backed tracking supports model registration. The model was successfully registered under the name `rvce-sentence-transformer`.

---

## 6. Verification and Test Results

The test suite consists of 44 automated tests across:
1. `tests/test_retrieval.py` (20 tests): TF-IDF chunking, boilerplate filters, deduplication, semantic retrieval schema, corpus embedding caching, hybrid fusion (linear & RRF), cross-encoder metadata preservation, error handling.
2. `tests/test_evaluation.py` (9 tests): Query result summarization, benchmark coverage, degradation thresholds, synthetic corpus reduction, labelled ranking metrics, MLflow logging, model reload smoke test.
3. `tests/test_answering.py` (5 tests): Evidence-based extraction, question matching, conservative abstention, source attribution.
4. `tests/test_api.py` (10 tests): FastAPI `/health`, `/ask`, root frontend serving, static assets, evidence requirements, 400 bad request handling.

**Test execution command**:
```powershell
.venv-hf-fix\Scripts\python.exe -m pytest tests/
```
**Actual Result**: `44 passed, 1 warning in 67.01s` (100% pass rate).

---

## 7. Known Limitations

1. **Unlabelled 40-Query Benchmark**: The primary benchmark CSV contains intent labels but lacks per-document relevance annotations. Coverage metrics must not be interpreted as factual answer correctness.
2. **Dense Model Specialization**: `all-MiniLM-L6-v2` is a zero-shot general-domain model and has not been fine-tuned on academic or Indian higher-education terminology.
3. **Cross-Encoder Cache**: Weights for `cross-encoder/ms-marco-MiniLM-L6-v2` were not cached locally in this environment; the system operated in verified graceful fallback mode.
4. **Local File Tracking Store**: MLflow tracking uses a local file store under `./mlruns` suitable for course evaluation, not a remote multi-user tracking server.
