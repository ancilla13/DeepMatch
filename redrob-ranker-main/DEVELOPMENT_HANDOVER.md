# Developer Handover & Status Summary: Scoring & Ranking Pipeline

This document is a standalone handover summary for **Person 3** to proceed with the implementation of the ranking module ([rank.py](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/rank.py)) and the web application interface ([app.py](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/app.py)). 

---

## 📌 Current Project Status

| Phase | Component | Status | Owner | Description |
|---|---|---|---|---|
| **1** | **Data Preprocessing** ([prepare.py](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/prepare.py)) | ✅ Complete | Person 1 | Data extraction, feature engineering, and embedding computation. |
| **2** | **Scoring Engine** ([scorer.py](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py)) | ✅ Complete | Person 2 | Modular intelligence scoring libraries, strict gates, and self-tests. |
| **3** | **Ranking Orchestration** ([rank.py](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/rank.py)) | ❌ Pending | **Person 3** | Candidate deduplication, ranking loops, reasoning, and JSON export. |
| **4** | **Streamlit UI App** ([app.py](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/app.py)) | ❌ Pending | **Person 3** | Recruiter candidate comparison dashboard and reasoning panel. |

---

## 🏗️ Preprocessing Completed (`prepare.py`)

The preprocessing phase has successfully parsed the source datasets and produced intermediate precomputed files to avoid redundant GPU/CPU embedding generation during execution. 

### What Was Done
1. **Dataset Loading**: Parsed candidate data ([sample_candidates.json](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/dataset/[PUB]%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json)) and the job description document ([job_description.docx](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/dataset/[PUB]%20India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx)).
2. **Text Embeddings**: Generated candidate text blobs (headline + summary + role descriptions) and vectorized them using HuggingFace's `all-MiniLM-L6-v2` model (384-dimensional).
3. **Role-Level Embeddings**: Generated chronological lists of embeddings for every specific role in the candidate's career history to support trajectory similarity calculations.
4. **Feature Extraction**: Calculated structured variables (years of experience, graduation year, earliest job year, total months, overlapping dates, pre-graduation employment status, tenure mismatches, and Redrob platform signals).
5. **CPU Execution Verified**: Validated that all vector operations execute cleanly without requiring GPU hardware.

---

## 📂 Precomputed Artifacts Available (`precomputed/`)

All generated artifacts are saved in the [precomputed/](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/precomputed/) directory:

* 📄 **[candidate_embeddings.npy](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/precomputed/candidate_embeddings.npy)**: 2D NumPy array of shape `(N, 384)` containing dense vector embeddings for all candidates.
* 📄 **[jd_embedding.npy](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/precomputed/jd_embedding.npy)**: 1D NumPy array of shape `(384,)` containing the job description's text embedding.
* 📄 **[candidate_ids.txt](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/precomputed/candidate_ids.txt)**: Text file containing unique candidate IDs corresponding exactly to the row indices in the embeddings matrix.
* 📄 **[candidate_features.json](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/precomputed/candidate_features.json)**: JSON dictionary mapping `candidate_id` to its extracted candidate features (for disqualification, honeypot detection, etc.).
* 📄 **[role_embeddings.json](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/precomputed/role_embeddings.json)**: JSON dictionary mapping `candidate_id` to a list of role embeddings (ordered newest to oldest) for career trajectory tracking.

---

## 🧠 Scoring Engine Completed (`scorer.py`)

The scoring intelligence library is fully implemented, verified, and side-effect-free (no print statements, no hardcoded paths, no file I/O). The validation self-test suite executes cleanly with **59 out of 59 checks passing**.

### Scoring Weights Configuration
* **Semantic Fit**: `30%` (Cosine similarity of candidate text blob vs. JD embedding)
* **JD Must-Have Keyword Coverage**: `20%` (4 categories: dense retrieval, vector databases, Python, evaluation frameworks)
* **platform Behavioral Signals**: `20%` (outreach response, trust verifications, GitHub activity, active search bonus, inactivity penalty)
* **Career Trajectory alignment**: `15%` (decay-weighted cosine similarity of past roles vs. JD)
* **Skill Authenticity**: `15%` (plausibility analysis of skill level vs. practice months, assessment boosts, and peer endorsements)

---

### Exponentiated Functions Map

All core logic functions are fully documented and available for import from [scorer.py](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py):

| Function | Reference Link | Description |
|---|---|---|
| `semantic_score` | [scorer.py#L173](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L173) | Cosine similarity between candidate and JD embeddings in `[0.0, 1.0]`. |
| `is_disqualified` | [scorer.py#L211](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L211) | Returns `True` if any hard disqualification flag (e.g., no production, langchain-only, etc.) is triggered. |
| `disqualification_detail` | [scorer.py#L260](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L260) | Returns a detailed breakdown of checked disqualification flags. |
| `honeypot_score` | [scorer.py#L315](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L315) | Integrity flag count based on overlapping dates, pre-grad careers, duration mismatches, and inflated skill claims. |
| `jd_requirement_coverage` | [scorer.py#L371](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L371) | Returns fractional keyword coverage across must-have categories. |
| `skill_authenticity` | [scorer.py#L415](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L415) | Cross-validates skill levels against duration, assessments, and endorsements. |
| `trajectory_score` | [scorer.py#L535](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L535) | Recency decay-weighted trajectory alignment score. |
| `behavioral_score` | [scorer.py#L591](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L591) | Normalizes response rate, verifications, and activity signals. |
| `title_shows_escalation` | [scorer.py#L690](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L690) | Helper that checks chronological title progression along ladders. |
| `anti_pattern_penalty` | [scorer.py#L749](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L749) | Returns penalties (up to `-0.25`) for title chasing and framework enthusiasm. |
| `flag_duplicates` | [scorer.py#L834](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L834) | Identifies duplicate profiles using memory-safe matrix chunking. |
| `final_score` | [scorer.py#L941](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L941) | Synthesizes composite scores, applying gates (hard zero on disqualification/honeypots) and penalties. |
| `generate_reasoning` | [scorer.py#L1018](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/scorer.py#L1018) | Generates deterministic, fact-driven natural language justifications for recruiters. |

> [!NOTE]
> To execute the internal validation tests on the scoring module, run the following PowerShell command:
> ```powershell
> $env:PYTHONUTF8=1; python scorer.py
> ```

---

## 🚀 Roadmap for Person 3: Tasks & Next Steps

Your goal is to build the ranking execution script ([rank.py](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/rank.py)) and the recruiter front-end dashboard ([app.py](file:///c:/Users/dsouz/Desktop/DeepMatch/redrob-ranker-main/app.py)).

### 1. Build the Ranking Engine (`rank.py`)
This script should process the entire dataset (or the sample dataset) and output a sorted JSON file containing final candidate scores and recruiter reasoning.

#### Recommended Execution Logic
```python
import os
import json
import numpy as np
from scorer import (
    flag_duplicates, is_disqualified, honeypot_score, semantic_score, 
    jd_requirement_coverage, skill_authenticity, trajectory_score, 
    behavioral_score, anti_pattern_penalty, final_score, generate_reasoning
)

# 1. Load precomputed files
candidate_ids = [line.strip() for line in open("precomputed/candidate_ids.txt")]
candidate_embeddings = np.load("precomputed/candidate_embeddings.npy")
candidate_features = json.load(open("precomputed/candidate_features.json"))
role_embeddings = json.load(open("precomputed/role_embeddings.json"))
jd_embedding = np.load("precomputed/jd_embedding.npy")

# Load raw dataset for original structures (skills list, career history, etc.)
raw_candidates = json.load(open("dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"))
candidate_map = {c["candidate_id"]: c for c in raw_candidates}

# 2. Run memory-safe duplicate detection
# Compute mock/real behavioral scores array aligned with candidate_ids to resolve duplicates
beh_scores = [behavioral_score(candidate_features[cid]["redrob_signals"]) for cid in candidate_ids]
duplicate_ids = flag_duplicates(candidate_embeddings, candidate_ids, behavioral_scores=beh_scores)

ranked_list = []

# 3. Iterative scoring loop
for i, cid in enumerate(candidate_ids):
    if cid in duplicate_ids:
        # Mark as duplicate and handle filtering or keep them with warning
        continue

    features = candidate_features[cid]
    raw_cand = candidate_map[cid]

    # Calculate individual components
    sem = semantic_score(candidate_embeddings[i], jd_embedding)
    disq = is_disqualified(raw_cand, features)
    honeypot_f = honeypot_score(raw_cand, features)
    cov = jd_requirement_coverage(" ".join(features["companies"] + features["industries"] + [raw_cand["profile"].get("summary", "")]), raw_cand.get("skills", []))
    auth = skill_authenticity(raw_cand.get("skills", []))
    
    role_embs = role_embeddings.get(cid, [])
    traj = trajectory_score(role_embs, jd_embedding)
    beh = beh_scores[i]
    penalty = anti_pattern_penalty(raw_cand.get("career_history", []), raw_cand["profile"].get("summary", ""))

    # Compute composite final score
    score = final_score(
        semantic_fit=sem,
        coverage=cov,
        trajectory=traj,
        authenticity=auth,
        behavioral=beh,
        anti_pattern_penalty_value=penalty,
        is_disq=disq,
        honeypot_flags=honeypot_f
    )

    # Generate fact-based recruiter reasoning
    scores_dict = {
        "final": score,
        "semantic_fit": sem,
        "coverage": cov,
        "trajectory": traj,
        "authenticity": auth,
        "behavioral": beh,
    }
    reasoning = generate_reasoning(raw_cand, scores_dict)

    ranked_list.append({
        "candidate_id": cid,
        "name": raw_cand["profile"].get("name", "Candidate"),
        "final_score": score,
        "breakdown": scores_dict,
        "is_disqualified": disq,
        "honeypot_flags": honeypot_f,
        "reasoning": reasoning
    })

# 4. Sort and export results
ranked_list.sort(key=lambda x: x["final_score"], reverse=True)
with open("ranked_candidates.json", "w", encoding="utf-8") as f:
    json.dump(ranked_list, f, indent=2)
```

> [!WARNING]
> Ensure that you implement duplicate detection correctly. `flag_duplicates` uses a cosine similarity threshold of `0.95`. Under duplicate scenarios, the candidate with the higher platform behavioral score must be kept, and the duplicates flagged for omission.

---

### 2. Build the Downstream dashboard (`app.py`)
Implement the Streamlit dashboard to load `ranked_candidates.json` and showcase matching results to recruitment teams.

#### UI Feature Checklist
* **Overview KPI Cards**: Display overall candidate pool statistics, number of active seekers, disqualification rate, and flags detected.
* **Match Score Leaderboard**: A clean, sortable data table rendering candidate names, final composite scores, semantic fit, and behavioral activity levels.
* **Quality Filter Controls**: Toggle buttons to hide disqualified profiles, filter out high-flag honeypot candidates, and slider filters for minimum experience years.
* **Granular Profile Detail View**: Clicking on a candidate should open an overlay/sidebar displaying:
  - Natural-language reasoning statement generated by `generate_reasoning`.
  - Highlighted MUST-HAVE skills coverage visual badge indicators.
  - Line chart illustrating chronological trajectory alignment.
  - Profile integrity audits (breakdown of checks passing or failing).
