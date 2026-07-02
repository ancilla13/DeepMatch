import os
import json
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
from datetime import datetime

DATA_DIR = r"dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
PRECOMPUTED_DIR = "precomputed"

# Helper Functions
def extract_grad_year(candidate):
    years = []
    for edu in candidate.get("education", []):
        end_year = edu.get("end_year")
        if end_year:
            years.append(end_year)
    return min(years) if years else None

def extract_earliest_job_year(candidate):
    years = []
    for role in candidate.get("career_history", []):
        start_date = role.get("start_date")
        if start_date:
            years.append(int(start_date[:4]))
    return min(years) if years else None

def total_career_months(candidate):
    return sum(
        role.get("duration_months", 0)
        for role in candidate.get("career_history", [])
    )

def started_before_graduating(candidate):
    grad_year = extract_grad_year(candidate)
    if grad_year is None:
        return False
    violations = 0
    for role in candidate.get("career_history", []):
        start_date = role.get("start_date")
        if start_date:
            start_year = int(start_date[:4])
            if start_year < grad_year - 1:
                violations += 1
    return violations > 0

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m")
    except:
        return None

def has_overlapping_dates(candidate):
    roles = candidate.get("career_history", [])
    intervals = []
    for role in roles:
        start = parse_date(role.get("start_date"))
        end = parse_date(role.get("end_date"))
        if role.get("is_current"):
            end = datetime.now()
        if start and end:
            intervals.append((start, end))
    intervals.sort()
    for i in range(len(intervals) - 1):
        current_end = intervals[i][1]
        next_start = intervals[i + 1][0]
        if next_start < current_end:
            return True
    return False

def duration_mismatch_count(candidate):
    mismatches = 0
    for role in candidate.get("career_history", []):
        start = parse_date(role.get("start_date"))
        end = parse_date(role.get("end_date"))
        if role.get("is_current"):
            end = datetime.now()
        if not start or not end:
            continue
        actual_months = (end.year - start.year) * 12 + (end.month - start.month)
        declared = role.get("duration_months", 0)
        if abs(actual_months - declared) > 2:
            mismatches += 1
    return mismatches

def read_docx(path):
    doc = Document(path)
    return "\n".join(para.text for para in doc.paragraphs)

def build_candidate_blob(candidate):
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    role_descriptions = []
    for role in candidate.get("career_history", []):
        role_descriptions.append(role.get("description", ""))
    return " ".join([headline, summary] + role_descriptions)

# Core Pipeline Functions for API/Library Use
def extract_candidate_features_dict(candidates):
    candidate_features = {}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        candidate_features[cid] = {
            "yoe": candidate["profile"].get("years_of_experience", 0),
            "grad_year": extract_grad_year(candidate),
            "earliest_job_year": extract_earliest_job_year(candidate),
            "total_career_months": total_career_months(candidate),
            "has_overlapping_dates": has_overlapping_dates(candidate),
            "started_before_graduating": started_before_graduating(candidate),
            "duration_mismatch_count": duration_mismatch_count(candidate),
            "skills": candidate.get("skills", []),
            "redrob_signals": candidate.get("redrob_signals", {}),
            "current_title": candidate["profile"].get("current_title", ""),
            "current_company": candidate["profile"].get("current_company", ""),
            "companies": [role.get("company", "") for role in candidate.get("career_history", [])],
            "industries": [role.get("industry", "") for role in candidate.get("career_history", [])]
        }
    return candidate_features

def run_prepare_pipeline(candidates, jd_text, model=None):
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    # 1. Embed JD
    jd_embedding = model.encode(jd_text, convert_to_numpy=True)
    
    # 2. Embed Candidates
    candidate_blobs = [build_candidate_blob(c) for c in candidates]
    candidate_embeddings = model.encode(
        candidate_blobs,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    
    # 3. Embed Roles
    role_embeddings = {}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        roles = sorted(
            candidate.get("career_history", []),
            key=lambda x: x.get("start_date", ""),
            reverse=True
        )
        role_texts = []
        for role in roles:
            role_text = role.get("title", "") + " " + role.get("description", "")
            role_texts.append(role_text)
        
        if role_texts:
            embs = model.encode(role_texts, convert_to_numpy=True)
            role_embeddings[cid] = [emb.tolist() for emb in embs]
            
    # 4. Extract Features
    features = extract_candidate_features_dict(candidates)
    
    candidate_ids = [c["candidate_id"] for c in candidates]
    
    return {
        "candidate_ids": candidate_ids,
        "candidate_embeddings": candidate_embeddings,
        "jd_embedding": jd_embedding,
        "role_embeddings": role_embeddings,
        "candidate_features": features
    }

def main():
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
    CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates.jsonl")
    
    candidates = []
    skipped = 0
    with open(CANDIDATES_FILE, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1

    if skipped:
        print(f"  [WARN] Skipped {skipped} malformed lines in {CANDIDATES_FILE}")
    print(f"Loaded {len(candidates)} candidates from {CANDIDATES_FILE}")
    
    jd_text = read_docx(os.path.join(DATA_DIR, "job_description.docx"))
    print("JD loaded")
    
    print("Running pipeline...")
    results = run_prepare_pipeline(candidates, jd_text)
    
    # Save artifacts
    np.save(os.path.join(PRECOMPUTED_DIR, "jd_embedding.npy"), results["jd_embedding"])
    np.save(os.path.join(PRECOMPUTED_DIR, "candidate_embeddings.npy"), results["candidate_embeddings"])
    
    with open(os.path.join(PRECOMPUTED_DIR, "candidate_ids.txt"), "w", encoding="utf-8") as f:
        for cid in results["candidate_ids"]:
            f.write(cid + "\n")
            
    with open(os.path.join(PRECOMPUTED_DIR, "role_embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(results["role_embeddings"], f)
        
    with open(os.path.join(PRECOMPUTED_DIR, "candidate_features.json"), "w", encoding="utf-8") as f:
        json.dump(results["candidate_features"], f, indent=2)
        
    print("All artifacts saved successfully.")

if __name__ == "__main__":
    main()