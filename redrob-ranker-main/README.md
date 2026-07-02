# DeepMatch 🎯
**AI-Powered Candidate Discovery & Ranking Platform**

DeepMatch is a deterministic, local-first candidate intelligence pipeline and dashboard designed to help recruiters process massive talent pools with pinpoint accuracy. 

By combining NLP semantic matching with strict profile validation rules, DeepMatch reduces manual screening effort, detects fraudulent profiles (honeypots), and ranks the exact best-fit candidates for a given role—all while remaining 100% local, secure, and blazingly fast.

---

## 🚀 The Problem vs. Our Solution

**The Problem:** Recruiters receive massive, unstructured talent pools (like 100k+ candidates in a JSONL file). Traditional keyword matchers are too rigid (missing that "GCP" and "Google Cloud Platform" are the same), while generative LLMs are too slow, expensive, and prone to "hallucinating" ranks. Furthermore, recruiters waste time on fake resumes (honeypots) or duplicate profiles.

**The DeepMatch Solution:** A lightweight pipeline using deep semantic embeddings (SentenceTransformers) for contextual understanding, constrained by rigorous mathematical rules and deterministic data validation. It outputs an exact, explainable rank and visualizes it on a premium enterprise dashboard.

---

## ✨ Key Features

* **High-Accuracy Semantic Matching:** Uses local, CPU-friendly `SentenceTransformers` (`all-MiniLM-L6-v2`) to evaluate deep contextual fits. By using Cosine Similarity, the system understands conceptually related skills without rigid keyword constraints.
* **Honeypot & Fraud Detection:** Scans profiles for impossibilities (e.g., mastering an expert skill in 2 months, impossible date overlaps). Each red flag adds a penalty. Profiles with ≥ 2 flags are instantly disqualified to protect the recruiter's pipeline.
* **Twin Deduplication Matrix:** An O(N) memory-safe similarity matrix efficiently detects twin/duplicate/cloned candidate profiles (≥ 95% similarity) and silently removes the lesser clone.
* **Explainable Decisions (Zero Hallucinations):** Every candidate receives a clear, auto-generated "Reasoning" string. Because the rank is pure math, hallucinations are technically impossible.
* **Premium Glassmorphism Dashboard:** A sleek, fully responsive dark-mode UI built with Flask and Vanilla JS. Features interactive Top N filtering, score distributions, and persistent local state for manual candidate Accept/Reject tracking.

---

## 🧠 Scoring Methodology & Formulas

The final rank isn't guessed by an AI; it is mathematically calculated exactly as:
> `Final Score = (0.50 × Semantic_Match) + (0.20 × Behavioral_Score) + (0.15 × Trajectory_Score) + (0.15 × Authenticity_Score)`

*   **Semantic Match (50%):** Calculated using exact **Cosine Similarity**: `(A · B) / (||A|| * ||B||)`
*   **Behavioral Signals (20%):** Recruiter responsiveness, active job-seeking status.
*   **Career Trajectory (15%):** Growth speed, recent promotions.
*   **Authenticity (15%):** Profile consistency, verified contacts.

**Honeypot Penalty:** If a candidate triggers a red flag, their score is penalized: `Final Score = Final Score × (0.50 ^ honeypot_flags)`

---

## 🏗️ System Architecture & Tech Stack

The architecture is designed to be **Local, Secure, and Scalable**.

*   **Python 3.10 (Backend):** Core data processing and orchestration.
*   **SentenceTransformers / HuggingFace (AI Layer):** Chosen for providing state-of-the-art NLP embeddings in a lightweight, local, open-source package. Completely CPU bound (no GPU required).
*   **NumPy / Pandas (Data Handling):** Chosen for blazingly fast mathematical operations (Cosine Similarity, Deduplication Matrix) and `.xlsx` generation without the overhead of heavy vector databases.
*   **Flask (API Server):** Chosen for spinning up a lightweight, robust local web server to serve the API and UI dynamically.
*   **Vanilla JS & CSS (Frontend):** Modern Glassmorphism aesthetic chosen to build a premium, highly responsive enterprise dashboard without frontend framework bloat.

---

## 📂 Project Structure

```text
DeepMatch/
├── dataset/                     # Contains candidates.jsonl and job_description.docx
├── frontend/                    # Vanilla JS, HTML, CSS for the premium dashboard
├── precomputed/                 # Storage for generated NumPy embeddings and features
├── rank.py                      # Main pipeline orchestrator (CLI)
├── scorer.py                    # Intelligence engine (rules, formulas, deduplication)
├── server.py                    # Flask REST API and UI server
├── prepare.py                   # Data ingestion and preprocessing scripts
└── README.md
```

---

## 🛠️ Setup & Installation

**Prerequisites:** Python 3.10+

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pre-computation Pipeline (Offline)**
   Process the dataset, parse the JD, run the deduplication, and score the candidates:
   ```bash
   python rank.py
   ```
   *This outputs `submission.csv` (for hackathon judging) and `ranked_candidates.json` (for the dashboard).*

3. **Launch the Dashboard**
   ```bash
   python server.py
   ```
   *Navigate to `http://localhost:5000` to view the UI.*

---

## 🔒 Security & Hardware Constraints

* **CPU-Only Execution:** 100% compliant with standard hardware limits. Zero GPU or external cloud compute required (the `all-MiniLM-L6-v2` model is only ~80MB).
* **Data Privacy:** All data processing, embedding generation, and scoring occurs locally. **Zero API Costs.** No external APIs (like OpenAI) are called, ensuring candidate PII remains completely secure and data privacy breaches are impossible.

---

## 🏆 Hackathon Submission

The final output file generated for judges is **`submission.csv`** (or `submission.xlsx`), formatted perfectly according to competition guidelines. The `server.py` dashboard acts as an interactive visualization for the generated insights.
