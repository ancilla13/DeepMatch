import os
import json
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer

DATA_DIR = r"dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"

PRECOMPUTED_DIR = "precomputed"

os.makedirs(PRECOMPUTED_DIR, exist_ok=True)




# --------------------------
# Load candidates
# --------------------------

with open(
    os.path.join(DATA_DIR, "sample_candidates.json"),
    encoding="utf-8"
) as f:
    candidates = json.load(f)

print(f"Loaded {len(candidates)} candidates")
print(
    candidates[0]["career_history"][0]["start_date"]
)

from datetime import datetime


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
        return datetime.strptime(
            date_str,
            "%Y-%m"
        )
    except:
        return None


def has_overlapping_dates(candidate):

    roles = candidate.get(
        "career_history",
        []
    )

    intervals = []

    for role in roles:

        start = parse_date(
            role.get("start_date")
        )

        end = parse_date(
            role.get("end_date")
        )

        if role.get("is_current"):
            end = datetime.now()

        if start and end:
            intervals.append(
                (start, end)
            )

    intervals.sort()

    for i in range(
        len(intervals) - 1
    ):

        current_end = intervals[i][1]

        next_start = intervals[i + 1][0]

        if next_start < current_end:
            return True

    return False

def duration_mismatch_count(candidate):

    mismatches = 0

    for role in candidate.get(
        "career_history",
        []
    ):

        start = parse_date(
            role.get("start_date")
        )

        end = parse_date(
            role.get("end_date")
        )

        if role.get("is_current"):
            end = datetime.now()

        if not start or not end:
            continue

        actual_months = (
            (end.year - start.year) * 12
            + (end.month - start.month)
        )

        declared = role.get(
            "duration_months",
            0
        )

        if abs(actual_months - declared) > 2:
            mismatches += 1

    return mismatches

# --------------------------
# Read Job Description
# --------------------------

def read_docx(path):
    doc = Document(path)
    return "\n".join(
        para.text
        for para in doc.paragraphs
    )


jd_text = read_docx(
    os.path.join(DATA_DIR, "job_description.docx")
)

print("JD loaded")
print(jd_text[:300])


# --------------------------
# Candidate Blob Builder
# --------------------------

def build_candidate_blob(candidate):

    profile = candidate.get("profile", {})

    headline = profile.get("headline", "")
    summary = profile.get("summary", "")

    role_descriptions = []

    for role in candidate.get("career_history", []):

        role_descriptions.append(
            role.get("description", "")
        )

    return " ".join(
        [headline, summary] + role_descriptions
    )


candidate_blobs = []

for candidate in candidates:

    blob = build_candidate_blob(candidate)

    candidate_blobs.append(blob)

print("Built candidate blobs")


# --------------------------
# Load Embedding Model
# --------------------------

print("Loading MiniLM model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

print("Model loaded")


# --------------------------
# JD Embedding
# --------------------------

jd_embedding = model.encode(
    jd_text,
    convert_to_numpy=True
)

np.save(
    os.path.join(
        PRECOMPUTED_DIR,
        "jd_embedding.npy"
    ),
    jd_embedding
)

print("JD embedding saved")


# --------------------------
# Candidate Embeddings
# --------------------------

candidate_embeddings = model.encode(
    candidate_blobs,
    batch_size=32,
    convert_to_numpy=True,
    show_progress_bar=True
)

np.save(
    os.path.join(
        PRECOMPUTED_DIR,
        "candidate_embeddings.npy"
    ),
    candidate_embeddings
)

print("Candidate embeddings saved")


# --------------------------
# Candidate IDs
# --------------------------

with open(
    os.path.join(
        PRECOMPUTED_DIR,
        "candidate_ids.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    for candidate in candidates:

        f.write(
            candidate["candidate_id"] + "\n"
        )

print("Candidate IDs saved")


# --------------------------
# Verify
# --------------------------

print("\nEmbedding Shape:")
print(candidate_embeddings.shape)

print("\nDone.")



role_embeddings = {}

for candidate in candidates:

    cid = candidate["candidate_id"]

    roles = sorted(
        candidate["career_history"],
        key=lambda x:
            x["start_date"],
        reverse=True
    )

    role_texts = []

    for role in roles:

        role_text = (
            role.get("title", "")
            + " "
            + role.get(
                "description",
                ""
            )
        )

        role_texts.append(
            role_text
        )

    if len(role_texts) > 0:

        embeddings = model.encode(
            role_texts,
            convert_to_numpy=True
        )

        role_embeddings[cid] = [
            emb.tolist()
            for emb in embeddings
        ]

with open(
    os.path.join(
        PRECOMPUTED_DIR,
        "role_embeddings.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        role_embeddings,
        f
    )
print("Role embeddings saved")
candidate_features = {}
for candidate in candidates:

    cid = candidate[
        "candidate_id"
    ]

    candidate_features[cid] = {

        "yoe":
        candidate["profile"].get(
            "years_of_experience",
            0
        ),

        "grad_year":
        extract_grad_year(
            candidate
        ),

        "earliest_job_year":
        extract_earliest_job_year(
            candidate
        ),

        "total_career_months":
        total_career_months(
            candidate
        ),

        "has_overlapping_dates":
        has_overlapping_dates(
            candidate
        ),

        "started_before_graduating":
        started_before_graduating(
            candidate
        ),

        "duration_mismatch_count":
        duration_mismatch_count(
            candidate
        ),

        "skills":
        candidate.get(
            "skills",
            []
        ),

        "redrob_signals":
        candidate.get(
            "redrob_signals",
            {}
        ),

        "current_title":
        candidate["profile"].get(
            "current_title",
            ""
        ),

        "current_company":
        candidate["profile"].get(
            "current_company",
            ""
        ),

        "companies":
        [
            role.get(
                "company",
                ""
            )
            for role in candidate.get(
                "career_history",
                []
            )
        ],

        "industries":
        [
            role.get(
                "industry",
                ""
            )
            for role in candidate.get(
                "career_history",
                []
            )
        ]
    }

with open(
    os.path.join(
        PRECOMPUTED_DIR,
        "candidate_features.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        candidate_features,
        f,
        indent=2
    )

print("Candidate features saved")