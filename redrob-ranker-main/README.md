# Person 1 – Data Preparation & Feature Engineering

## Overview

This module is responsible for preparing candidate and job description data for the AI-powered candidate ranking system.

The goal of Person 1 is to transform raw candidate profiles and job descriptions into structured and machine-readable artifacts that can be used by the scoring and ranking modules.

---

# Responsibilities

The preparation pipeline performs the following tasks:

1. Load candidate profiles
2. Read the job description
3. Generate semantic embeddings
4. Extract structured candidate features
5. Detect profile inconsistencies
6. Generate role-level embeddings
7. Save all precomputed artifacts

The generated outputs are consumed by:

* Person 2 – Scoring Engine (`scorer.py`)
* Person 3 – Ranking Pipeline (`rank.py`)

---

# Dataset Structure

```text
dataset/
└── [PUB] India_runs_data_and_ai_challenge/
    └── India_runs_data_and_ai_challenge/
        ├── candidate_schema.json
        ├── candidates.jsonl
        ├── sample_candidates.json
        ├── job_description.docx
        ├── README.docx
        ├── redrob_signals_doc.docx
        ├── submission_spec.docx
        └── ...
```

---

# Candidate Schema

Each candidate contains:

```json
{
  "candidate_id": "...",
  "profile": {},
  "career_history": [],
  "education": [],
  "skills": [],
  "certifications": [],
  "languages": [],
  "redrob_signals": {}
}
```

---

# Job Description Processing

The job description is loaded from:

```text
job_description.docx
```

The text is extracted using:

```python
python-docx
```

and converted into a single text document.

---

# Candidate Text Blob Construction

Each candidate is converted into a text blob:

```text
headline
+
summary
+
all role descriptions
```

Example:

```text
Senior Machine Learning Engineer
Experienced AI engineer...
Built recommendation systems...
Developed NLP pipelines...
...
```

This blob represents the candidate's professional profile.

---

# Embedding Generation

Model Used:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding Dimension:

```text
384
```

Generated Embeddings:

1. Job Description Embedding
2. Candidate Profile Embeddings
3. Role-Level Embeddings

---

# Candidate Features Extracted

For every candidate, the following features are generated:

## Experience Features

* years_of_experience
* earliest_job_year
* total_career_months

## Education Features

* graduation_year

## Career Features

* companies worked
* industries worked
* current title
* current company

## Skills Features

* skill names
* proficiency levels
* endorsements
* duration

## Behavioral Signals

All Redrob behavioral signals are preserved for downstream scoring.

---

# Profile Integrity Checks

The pipeline performs several consistency checks.

## Started Before Graduation

Checks whether professional experience began significantly before graduation.

```python
started_before_graduating
```

---

## Career Duration Validation

Compares:

```text
Declared Duration
vs
Actual Duration
```

from start and end dates.

Output:

```python
duration_mismatch_count
```

---

## Overlapping Roles

Detects overlapping employment periods.

Output:

```python
has_overlapping_dates
```

---

# Generated Files

The following files are written to:

```text
precomputed/
```

## 1. Job Description Embedding

```text
jd_embedding.npy
```

Contains:

```text
384-dimensional JD vector
```

---

## 2. Candidate Embeddings

```text
candidate_embeddings.npy
```

Contains:

```text
N × 384 matrix
```

where N = number of candidates.

---

## 3. Candidate IDs

```text
candidate_ids.txt
```

Maintains ordering consistency between IDs and embeddings.

---

## 4. Role Embeddings

```text
role_embeddings.json
```

Contains embeddings for every role in a candidate's career history.

Used for:

```text
Career trajectory scoring
Promotion analysis
Role progression analysis
```

---

## 5. Candidate Features

```text
candidate_features.json
```

Contains all structured candidate metadata and integrity signals.

Example:

```json
{
  "yoe": 6.9,
  "grad_year": 2020,
  "earliest_job_year": 2019,
  "total_career_months": 82,
  "has_overlapping_dates": false,
  "started_before_graduating": false,
  "duration_mismatch_count": 0
}
```

---

# Output Directory

```text
precomputed/
├── candidate_embeddings.npy
├── candidate_features.json
├── candidate_ids.txt
├── jd_embedding.npy
└── role_embeddings.json
```

---

# Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

Main packages:

```text
numpy
sentence-transformers
python-docx
json
os
datetime
```

---

# Running the Pipeline

Execute:

```bash
python prepare.py
```

Expected output:

```text
Loaded candidates
JD loaded
Built candidate blobs
Model loaded
JD embedding saved
Candidate embeddings saved
Role embeddings saved
Candidate features saved
Done
```

---

# Handoff to Person 2

The following files must be provided to the scoring module:

```text
jd_embedding.npy
candidate_embeddings.npy
candidate_ids.txt
role_embeddings.json
candidate_features.json
```

These artifacts are used to compute:

* Semantic Match Score
* Coverage Score
* Trajectory Score
* Authenticity Score
* Behavioral Score
* Final Candidate Score

---

# Author

Person 1 – Data Preparation & Feature Engineering

Responsibilities:

* Data ingestion
* Embedding generation
* Feature extraction
* Candidate integrity checks
* Precomputed artifact generation
