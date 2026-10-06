# Campus AI Assistant — Phase 1
### ML Development & Reproducibility

An AI-powered Campus Information Assistant designed to understand student queries and route them to the appropriate campus information source. The Phase 1 implementation focuses on developing and evaluating a reproducible **Campus Query Intent Classification** model as the core ML component.

---

## 1. Problem Statement

Students often need information about academics, administration, hostels, facilities, fees, events, placements, and campus experiences. These queries may be phrased in many different ways and may require information from different sources.

The Phase 1 objective is to develop a supervised machine learning model that automatically identifies the **intent of a student's campus-related query**.

### Example

> "How do I apply for hostel accommodation?"

**Predicted Intent:** `HOSTEL`

> "When are the semester exams?"

**Predicted Intent:** `ACADEMICS`

> "How is the mess food?"

**Predicted Intent:** `STUDENT_EXPERIENCE`

The intent classifier will later act as the routing component of the complete Campus AI Assistant.

---

## 2. Phase 1 Objective

The objectives of Phase 1 are:

- Acquire and document a labeled campus-query dataset.
- Define and validate the query-intent taxonomy.
- Perform exploratory data analysis.
- Clean and preprocess the dataset.
- Engineer text features using TF-IDF.
- Develop baseline classification models.
- Compare model performance using standard evaluation metrics.
- Track experiments using MLflow.
- Version the dataset using DVC.
- Maintain the project using Git/GitHub.
- Ensure the training process is reproducible.

---

## 3. ML Task

### Task

**Multiclass Text Classification**

### Input

A natural-language student query.

### Output

One of the predefined campus-information intent classes.

### Initial Intent Classes

| Intent | Description |
|---|---|
| `ACADEMICS` | Courses, examinations, attendance, timetable, electives, academic regulations |
| `ADMINISTRATION` | Certificates, applications, documents, offices, administrative procedures |
| `HOSTEL` | Hostel accommodation, rules, timings, facilities and related queries |
| `FACILITIES` | Library, laboratories, medical facilities, sports, Wi-Fi and campus infrastructure |
| `FEES_FINANCE` | Tuition fees, hostel fees, scholarships and payment-related queries |
| `EVENTS_CLUBS` | Clubs, events, workshops, hackathons and student activities |
| `PLACEMENTS` | Placements, internships, recruitment and career-related campus information |
| `STUDENT_EXPERIENCE` | Student opinions and experiences regarding campus life, food, facilities, faculty and similar topics |
| `OUT_OF_SCOPE` | Queries unrelated to campus information |

---

## 4. Dataset

The primary ML dataset consists of labeled student-style campus queries.

### Dataset Schema

```text
query,label
"When are the semester exams?",ACADEMICS
"How do I apply for hostel accommodation?",HOSTEL
"Where is the central library?",FACILITIES
"How can I obtain a bonafide certificate?",ADMINISTRATION
"How much is the hostel fee?",FEES_FINANCE
"How is the campus mess?",STUDENT_EXPERIENCE
"How are placement opportunities?",PLACEMENTS
"How do I join a student club?",EVENTS_CLUBS
"What is the capital of France?",OUT_OF_SCOPE
```

### Target Dataset Size

Initial target: **3,000–5,000 labeled queries**.

The dataset will be maintained with reasonable class balance to prevent the classifier from being dominated by a single intent.

### Data Sources

The dataset may contain:

1. Official campus information used to identify relevant information categories.
2. Manually authored student-style queries.
3. Permitted public community discussions used to identify realistic query patterns.
4. Manually verified/synthetic variations of campus queries.

Community-generated content will **not be treated as authoritative campus information**. It is used primarily to capture realistic student language and experiences.

---

## 5. Data Processing Pipeline

```text
Raw Data
   ↓
Data Validation
   ↓
Cleaning
   ↓
Duplicate Detection
   ↓
Label Validation
   ↓
Train / Validation / Test Split
   ↓
Feature Engineering
   ↓
Model Training
```

Data processing will include:

- Missing-value detection
- Duplicate detection
- Text normalization
- Label validation
- Query-length analysis
- Class-distribution analysis
- Train/validation/test splitting

---

## 6. Exploratory Data Analysis

The dataset will be analyzed for:

- Total number of samples
- Number of classes
- Class distribution
- Missing values
- Duplicate queries
- Query length distribution
- Most common terms
- Representative examples from each class
- Potential class imbalance

EDA will be documented in:

```text
notebooks/01_eda.ipynb
```

---

## 7. Feature Engineering

The initial feature representation will use **TF-IDF (Term Frequency–Inverse Document Frequency)**.

### Pipeline

```text
Student Query
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
Unigram + Bigram Features
     ↓
TF-IDF Vectorization
     ↓
Numerical Feature Matrix
```

The feature engineering stage will evaluate parameters such as:

- `ngram_range`
- `max_features`
- `min_df`
- TF-IDF normalization

---

## 8. Baseline Models

The following models will be evaluated:

### Baseline 1 — Majority Class

Predicts the most frequent class for every query.

This establishes a minimum performance benchmark.

### Baseline 2 — Logistic Regression

```text
TF-IDF → Logistic Regression → Intent
```

### Baseline 3 — Linear Support Vector Machine

```text
TF-IDF → Linear SVM → Intent
```

Additional classical models may be evaluated if required.

---

## 9. Model Evaluation

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- **Macro F1-score**
- Weighted F1-score
- Confusion Matrix

### Primary Metric

**Macro F1-score** will be used as the primary metric because it gives equal importance to each intent class.

The final model will be selected based on validation performance while considering generalization and class-wise performance.

---

## 10. Experiment Tracking — MLflow

MLflow will be used to track model development experiments.

Each experiment will record:

### Parameters

- Model type
- TF-IDF configuration
- Hyperparameters
- Random seed
- Dataset version

### Metrics

- Accuracy
- Precision
- Recall
- Macro F1
- Weighted F1

### Artifacts

- Trained model
- Confusion matrix
- Evaluation reports
- Feature configuration

Example experiment:

```text
Experiment: campus_query_classification

Run 1
TF-IDF + Logistic Regression

Run 2
TF-IDF + Linear SVM

Run 3
Alternative baseline
```

---

## 11. Dataset Versioning — DVC

DVC will be used to version the dataset independently from Git.

```text
Dataset v1
    ↓
Model Experiment v1

Dataset v2
    ↓
Model Experiment v2
```

This allows each model experiment to be associated with a specific dataset version and ensures reproducibility.

---

## 12. Reproducibility

The project will maintain reproducibility through:

- Git version control
- DVC dataset versioning
- MLflow experiment tracking
- Fixed random seeds
- Dependency versioning
- Configuration files
- Documented preprocessing
- Versioned model artifacts

A future developer should be able to reproduce the Phase 1 experiment using the repository and the corresponding dataset version.

---

## 13. Repository Structure

```text
campus-ai-assistant/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
│
├── models/
│
├── reports/
│
├── tests/
│
├── params.yaml
├── dvc.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 14. Phase 1 Deliverables

By the end of Phase 1, the repository should contain:

- [ ] Approved dataset definition
- [ ] Labeled campus-query dataset
- [ ] Dataset documentation
- [ ] Data validation and preprocessing
- [ ] Exploratory Data Analysis
- [ ] Feature engineering pipeline
- [ ] Majority-class baseline
- [ ] Logistic Regression baseline
- [ ] Linear SVM baseline
- [ ] Initial model evaluation
- [ ] Confusion matrix
- [ ] Git repository with contribution history
- [ ] DVC dataset versioning
- [ ] MLflow experiment tracking
- [ ] Reproducible training configuration
- [ ] Phase 1 results documented in README/report

---

## 15. Phase 1 Scope Boundary

Phase 1 focuses specifically on:

> **Dataset → EDA → Feature Engineering → Baseline Models → Evaluation → Git → DVC → MLflow → Reproducibility**

The following components are outside the primary Phase 1 scope and can be developed in later phases:

- Production conversational interface
- RAG pipeline
- Community sentiment analysis
- Advanced semantic retrieval
- LLM integration
- Docker containerization
- CI/CD
- Deployment
- Model monitoring
- Continuous retraining

---

## 16. Future System

The Phase 1 intent classifier will become one component of the complete Campus AI Assistant:

```text
                 Student Query
                      │
                      ▼
            Campus Intent Classifier
                      │
             ┌────────┴────────┐
             ▼                 ▼
      Official Sources     Community Sources
             │                 │
             └────────┬────────┘
                      ▼
             Retrieval / RAG
                      │
                      ▼
              Campus AI Assistant
                      │
                      ▼
             Source-backed Answer
```

Phase 1 therefore establishes the **ML foundation and reproducible experimentation pipeline** for the complete system.
