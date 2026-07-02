# DeepMatch 🎯
**AI-Powered Candidate Discovery & Ranking Platform**

DeepMatch is a deterministic, AI-driven candidate intelligence pipeline and dashboard designed to help recruiters process large talent pools with pinpoint accuracy. By combining NLP semantic matching with strict profile validation rules, DeepMatch reduces manual screening effort, detects fraudulent profiles (honeypots), and ranks the exact best-fit candidates for a given role.

## 🚀 Key Features

* **High-Accuracy Semantic Matching:** Uses local, CPU-friendly `SentenceTransformers (all-MiniLM-L6-v2)` to evaluate deep contextual fits between Candidate Skills and Job Descriptions without relying on paid APIs.
* **Honeypot & Fraud Detection:** Advanced logic parses profiles for impossibilities (e.g., conflicting dates, impossibly fast expert skill mastery) and penalizes or quarantines fake "Honeypot" resumes.
* **Twin Deduplication Matrix:** An O(N) memory-safe matrix efficiently detects twin/duplicate candidate profiles (≥ 95% similarity) and filters them out of the leaderboard.
* **Strict Eligibility Gates:** Deterministic logic handles strict parameters like Graduation Years (e.g., 2023-2024), Minimum Years of Experience, and recency of hands-on coding.
* **Multi-Factor Scoring Engine:** Combines Semantic Match (50%), Behavioral Signals (20%), Authenticity (15%), and Career Trajectory (15%) into an exact mathematical rank.
* **Premium Glassmorphism Dashboard:** A sleek, fully responsive dark-mode UI built with Flask and Vanilla JS. Features interactive Top N filtering, live Live Sandbox evaluations, and persistent local state for manual candidate Accept/Reject tracking.

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

## 🔒 Security & Hardware Constraints
* **CPU-Only Execution:** 100% compliant with standard hardware limits. Zero GPU or external cloud compute required.
* **Data Privacy:** All data processing, embedding generation, and scoring occurs locally. No external APIs (like OpenAI) are called, ensuring candidate PII remains secure.

## 🏆 Hackathon Submission
The final output file generated for judges is **`submission.csv`**, formatted perfectly according to competition guidelines. The `server.py` dashboard acts as an interactive visualization for the generated insights.
