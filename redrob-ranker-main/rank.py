"""
rank.py — Redrob Candidate Ranking Engine
==========================================
Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints:
    - CPU only, no network, ≤5 min wall-clock, ≤16 GB RAM
    - Requires precomputed/ folder (produced by prepare.py)
    - Outputs exactly 100 rows ranked best-fit first

Submission CSV columns: candidate_id, rank, score, reasoning
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

# ── scorer.py must be in the same directory ──────────────────────────────────
try:
    from scorer import (
        is_disqualified,
        honeypot_score,
        semantic_score,
        jd_requirement_coverage,
        skill_authenticity,
        trajectory_score,
        behavioral_score,
        anti_pattern_penalty,
        flag_duplicates,
        final_score,
        generate_reasoning,
    )
except ImportError as e:
    sys.exit(f"[ERROR] Could not import scorer.py: {e}\n"
             "Make sure scorer.py is in the same directory as rank.py.")

# ── Constants ─────────────────────────────────────────────────────────────────
PRECOMPUTED_DIR = "precomputed"
TOP_N = 100
DUPLICATE_THRESHOLD = 0.95   # matches scorer.py flag_duplicates default
HONEYPOT_HARD_THRESHOLD = 2  # honeypot_score >= 2 → hard zero (per spec)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> dict:
    """Load candidates from a JSON array or JSONL (one JSON object per line) into {candidate_id: record}."""
    candidate_map = {}
    
    # Try loading as a single JSON array first
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for record in data:
                    cid = record.get("candidate_id")
                    if cid is not None:
                        candidate_map[cid] = record
                print(f"  Loaded {len(candidate_map):,} candidates from JSON array in {path}")
                return candidate_map
    except Exception:
        # Fall back to JSONL format if JSON loading fails
        pass

    candidate_map = {}
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cid = record.get("candidate_id")
                if cid is None:
                    skipped += 1
                    continue
                candidate_map[cid] = record
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"  [WARN] Skipped {skipped} malformed lines in {path}")
    print(f"  Loaded {len(candidate_map):,} candidates from JSONL in {path}")
    return candidate_map


def load_precomputed(precomputed_dir: str):
    """Load all precomputed artifacts. Fails loudly if any are missing."""
    def _path(filename):
        return os.path.join(precomputed_dir, filename)

    required = [
        "candidate_ids.txt",
        "candidate_embeddings.npy",
        "jd_embedding.npy",
        "candidate_features.json",
        "role_embeddings.json",
    ]
    for fname in required:
        if not os.path.exists(_path(fname)):
            sys.exit(f"[ERROR] Missing precomputed file: {_path(fname)}\n"
                     "Run prepare.py first to generate the precomputed/ folder.")

    with open(_path("candidate_ids.txt"), "r", encoding="utf-8") as f:
        candidate_ids = [line.strip() for line in f if line.strip()]

    candidate_embeddings = np.load(_path("candidate_embeddings.npy"))
    jd_embedding = np.load(_path("jd_embedding.npy"))

    with open(_path("candidate_features.json"), "r", encoding="utf-8") as f:
        candidate_features = json.load(f)

    with open(_path("role_embeddings.json"), "r", encoding="utf-8") as f:
        role_embeddings = json.load(f)

    # Alignment check — catches silent row-order bugs immediately
    if len(candidate_ids) != candidate_embeddings.shape[0]:
        sys.exit(
            f"[ERROR] Alignment mismatch: candidate_ids.txt has {len(candidate_ids)} entries "
            f"but candidate_embeddings.npy has {candidate_embeddings.shape[0]} rows. "
            "Re-run prepare.py to regenerate consistent artifacts."
        )

    print(f"  candidate_ids.txt       : {len(candidate_ids):,} entries")
    print(f"  candidate_embeddings.npy: {candidate_embeddings.shape}")
    print(f"  jd_embedding.npy        : {jd_embedding.shape}")
    print(f"  candidate_features.json : {len(candidate_features):,} keys")
    print(f"  role_embeddings.json    : {len(role_embeddings):,} keys")

    return candidate_ids, candidate_embeddings, jd_embedding, candidate_features, role_embeddings


def validate_csv(path: str, candidate_ids_in_dataset: set, expected_count: int = TOP_N):
    """Run all format checks from Section 2.1 / Section 10 of the spec."""
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_cols = {"candidate_id", "rank", "score", "reasoning"}
    if not required_cols.issubset(set(reader.fieldnames or [])):
        errors.append(f"Missing columns. Expected: {required_cols}, got: {reader.fieldnames}")

    if len(rows) != expected_count:
        errors.append(f"Row count is {len(rows)}, must be exactly {expected_count}.")

    ranks_seen = set()
    ids_seen = set()
    prev_score = None

    for i, row in enumerate(rows):
        try:
            rank = int(row["rank"])
        except (ValueError, KeyError):
            errors.append(f"Row {i+1}: rank '{row.get('rank')}' is not an integer.")
            continue

        try:
            score = float(row["score"])
        except (ValueError, KeyError):
            errors.append(f"Row {i+1}: score '{row.get('score')}' is not a float.")
            score = None

        cid = row.get("candidate_id", "")
        reasoning = row.get("reasoning", "")

        if rank in ranks_seen:
            errors.append(f"Row {i+1}: duplicate rank {rank}.")
        else:
            ranks_seen.add(rank)

        if cid in ids_seen:
            errors.append(f"Row {i+1}: duplicate candidate_id '{cid}'.")
        else:
            ids_seen.add(cid)

        if cid not in candidate_ids_in_dataset:
            errors.append(f"Row {i+1}: candidate_id '{cid}' not found in source dataset.")

        if score is not None and prev_score is not None:
            if score > prev_score + 1e-9:   # allow tiny float rounding
                errors.append(
                    f"Row {i+1}: score {score} > previous score {prev_score} — "
                    "must be non-increasing."
                )
        if score is not None:
            prev_score = score

        if not reasoning or len(reasoning.strip()) < 10:
            errors.append(f"Row {i+1} (rank {rank}): reasoning is empty or too short.")

    expected_ranks = set(range(1, expected_count + 1))
    missing_ranks = expected_ranks - ranks_seen
    if missing_ranks:
        errors.append(f"Missing ranks: {sorted(missing_ranks)[:10]} ...")

    return errors


# ── Core Ranking Function for API/Library Use ───────────────────────────────

def run_ranking_pipeline(
    candidate_ids,
    candidate_embeddings,
    jd_embedding,
    candidate_features,
    role_embeddings,
    candidate_map,
    top_n=TOP_N,
    yoe_filter=0,
    open_to_work_filter=False,
    skills_filter=None
):
    """
    Core engine logic separated from I/O so it can be called in-memory.
    Supports basic filters before scoring to test "upload + filter" workflow.
    """
    # 1. Twin / duplicate detection
    behavioral_scores_list = []
    for cid in candidate_ids:
        features = candidate_features.get(cid, {})
        signals = features.get("redrob_signals", {})
        beh = behavioral_score(signals) if signals else 0.0
        behavioral_scores_list.append(beh)

    duplicate_ids = flag_duplicates(
        candidate_embeddings,
        candidate_ids,
        behavioral_scores=behavioral_scores_list,
        threshold=DUPLICATE_THRESHOLD,
    )

    results = []
    disq_count = 0
    honeypot_count = 0
    skipped_count = 0

    for i, cid in enumerate(candidate_ids):
        # Skip duplicates
        if cid in duplicate_ids:
            skipped_count += 1
            continue

        # Skip if raw profile not available
        if cid not in candidate_map:
            skipped_count += 1
            continue

        raw_cand = candidate_map[cid]
        features = candidate_features.get(cid, {})
        cand_emb = candidate_embeddings[i]

        # Apply basic pre-filters if requested
        if yoe_filter > 0:
            yoe = raw_cand.get("profile", {}).get("years_of_experience", 0.0)
            if yoe < yoe_filter:
                skipped_count += 1
                continue
                
        if open_to_work_filter:
            # Check open to work status in signals
            signals = features.get("redrob_signals", {})
            if not signals.get("open_to_work_flag", False):
                skipped_count += 1
                continue

        if skills_filter:
            cand_skills = {s.get("name", "").lower() for s in raw_cand.get("skills", [])}
            missing = [s for s in skills_filter if s.lower() not in cand_skills]
            if missing:
                skipped_count += 1
                continue

        # ── Gate 1: Hard disqualification ────────────────────────────────────
        disq = is_disqualified(raw_cand, features)
        if disq:
            disq_count += 1
            results.append({
                "candidate_id": cid,
                "final_score": 0.0,
                "reasoning": "",
                "is_disqualified": True,
                "honeypot_flags": 0,
                "breakdown": {
                    "final": 0.0,
                    "semantic_fit": 0.0,
                    "coverage": 0.0,
                    "trajectory": 0.0,
                    "authenticity": 0.0,
                    "behavioral": 0.0
                }
            })
            continue

        # ── Gate 2: Honeypot check ────────────────────────────────────────────
        h_score = honeypot_score(raw_cand, features)
        if h_score >= HONEYPOT_HARD_THRESHOLD:
            honeypot_count += 1
            results.append({
                "candidate_id": cid,
                "final_score": 0.0,
                "reasoning": "",
                "is_disqualified": False,
                "honeypot_flags": h_score,
                "breakdown": {
                    "final": 0.0,
                    "semantic_fit": 0.0,
                    "coverage": 0.0,
                    "trajectory": 0.0,
                    "authenticity": 0.0,
                    "behavioral": 0.0
                }
            })
            continue

        # ── Scoring components ────────────────────────────────────────────────
        sem = semantic_score(cand_emb, jd_embedding)

        # Build text blob for coverage: companies + industries + summary
        companies = features.get("companies", [])
        industries = features.get("industries", [])
        summary = raw_cand.get("profile", {}).get("summary", "")
        headline = raw_cand.get("profile", {}).get("headline", "")
        text_blob = " ".join(filter(None, companies + industries + [headline, summary]))

        skills = raw_cand.get("skills", [])
        cov = jd_requirement_coverage(text_blob, skills)

        role_embs = role_embeddings.get(cid, [])
        traj = trajectory_score(role_embs, jd_embedding)

        assessment_scores = {
            s["name"]: s.get("assessment_score")
            for s in skills
            if s.get("assessment_score") is not None
        } if skills else {}
        auth = skill_authenticity(skills, assessment_scores)

        signals = features.get("redrob_signals", {})
        beh = behavioral_score(signals) if signals else 0.0

        career_history = raw_cand.get("career_history", [])
        penalty = anti_pattern_penalty(career_history, text_blob)

        # ── Composite score ───────────────────────────────────────────────────
        score = final_score(
            semantic_fit=sem,
            coverage=cov,
            trajectory=traj,
            authenticity=auth,
            behavioral=beh,
            anti_pattern_penalty_value=penalty,
            is_disq=False,
            honeypot_flags=h_score,
        )

        # ── Reasoning ─────────────────────────────────────────────────────────
        scores_dict = {
            "final": score,
            "semantic_fit": sem,
            "coverage": cov,
            "trajectory": traj,
            "authenticity": auth,
            "behavioral": beh,
        }
        reasoning = generate_reasoning(raw_cand, scores_dict)

        results.append({
            "candidate_id": cid,
            "final_score": score,
            "reasoning": reasoning,
            "is_disqualified": False,
            "honeypot_flags": h_score,
            "breakdown": scores_dict,
        })

    # Sort results
    # Primary sort: final_score descending; tie-break: candidate_id ascending
    eligible = [r for r in results if not r["is_disqualified"] and r["honeypot_flags"] < HONEYPOT_HARD_THRESHOLD]
    eligible.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    top_candidates = eligible[:top_n]
    
    # Sort all results for dashboard display
    sorted_all_results = sorted(results, key=lambda x: (-x["final_score"], x["is_disqualified"], x["honeypot_flags"], x["candidate_id"]))

    return top_candidates, sorted_all_results, duplicate_ids, disq_count, honeypot_count, skipped_count


# ── Main CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl (100K profiles)")
    parser.add_argument("--out", required=True,
                        help="Output CSV path (e.g. submission.csv)")
    parser.add_argument("--precomputed", default=PRECOMPUTED_DIR,
                        help="Precomputed artifacts directory (default: precomputed/)")
    parser.add_argument("--top", type=int, default=TOP_N,
                        help="Number of candidates to output (default: 100)")
    args = parser.parse_args()

    wall_start = time.time()
    print("\n" + "="*60)
    print("  Redrob Candidate Ranker — rank.py")
    print("="*60)

    # ── Step 1: Load everything ───────────────────────────────────────────────
    print("\n[1/6] Loading precomputed artifacts ...")
    (
        candidate_ids,
        candidate_embeddings,
        jd_embedding,
        candidate_features,
        role_embeddings,
    ) = load_precomputed(args.precomputed)

    print("\n[2/6] Loading raw candidate profiles ...")
    candidate_map = load_candidates(args.candidates)

    # Warn if any precomputed ID is not in the raw dataset
    missing_in_raw = [cid for cid in candidate_ids if cid not in candidate_map]
    if missing_in_raw:
        print(f"  [WARN] {len(missing_in_raw)} precomputed IDs not found in "
              f"{args.candidates}. They will be skipped.")

    # ── Step 2-4: Run Ranking ─────────────────────────────────────────────────
    print("\n[3/6] Running ranking pipeline ...")
    t0 = time.time()
    top_candidates, sorted_all_results, duplicate_ids, disq_count, honeypot_count, skipped_count = run_ranking_pipeline(
        candidate_ids,
        candidate_embeddings,
        jd_embedding,
        candidate_features,
        role_embeddings,
        candidate_map,
        top_n=args.top
    )
    elapsed_scoring = time.time() - t0
    print(f"  Scoring complete in {elapsed_scoring:.1f}s")
    print(f"  Disqualified : {disq_count:,}")
    print(f"  Honeypots    : {honeypot_count:,}")
    print(f"  Twins skipped: {len(duplicate_ids):,}")

    # ── Honeypot rate self-check ──────────────────────────────────────────────
    honeypot_in_top = sum(1 for r in top_candidates if r["honeypot_flags"] >= 1)
    honeypot_rate = honeypot_in_top / len(top_candidates) if top_candidates else 0
    print(f"  Honeypot rate in top {args.top}: "
          f"{honeypot_in_top}/{len(top_candidates)} = {honeypot_rate:.1%}")
    if honeypot_rate > 0.10:
        print("  [WARN] Honeypot rate > 10% — re-tune scoring weights before submitting!")

    if len(top_candidates) < args.top:
        print(f"  [WARN] Only {len(top_candidates)} eligible candidates found "
              f"(need {args.top}). Check disqualification logic.")

    # ── Step 5: Write outputs (CSV and JSON) ──────────────────────────────────
    print(f"\n[6/6] Writing outputs to {args.out} ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 1. Write submission CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for rank, record in enumerate(top_candidates, start=1):
            writer.writerow({
                "candidate_id": record["candidate_id"],
                "rank": rank,
                "score": record["final_score"],
                "reasoning": record["reasoning"],
            })

    # 2. Write enriched ranked_candidates.json for the dashboard
    json_results = []
    for r in top_candidates:
        cid = r["candidate_id"]
        raw_cand = candidate_map.get(cid, {})
        profile = raw_cand.get("profile", {})
        
        json_results.append({
            "candidate_id": cid,
            "name": profile.get("anonymized_name", cid),
            "final_score": r["final_score"],
            "reasoning": r["reasoning"],
            "is_disqualified": r["is_disqualified"],
            "honeypot_flags": r["honeypot_flags"],
            "breakdown": r.get("breakdown", {}),
            "profile": {
                "current_title": profile.get("current_title", "—"),
                "current_company": profile.get("current_company", "—"),
                "years_of_experience": profile.get("years_of_experience", 0.0),
            },
            "yoe": profile.get("years_of_experience", 0.0),
        })
        
    json_out_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "ranked_candidates.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved enriched JSON dashboard data to: {json_out_path}")

    # ── Format validation (built-in) ──────────────────────────────────────────
    print("\n  Running format validation ...")
    errors = validate_csv(args.out, set(candidate_map.keys()), expected_count=len(top_candidates))
    if errors:
        print(f"\n  [FAIL] {len(errors)} validation error(s):")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)
    else:
        print("  [PASS] All format checks passed (100 rows, ranks 1-100, "
              "non-increasing scores, no duplicates)")

    # ── Final timing summary ──────────────────────────────────────────────────
    wall_elapsed = time.time() - wall_start
    print("\n" + "="*60)
    print(f"  Done. Output: {args.out}")
    print(f"  Total wall-clock time : {wall_elapsed:.1f}s "
          f"({'[PASS] within 5-min limit' if wall_elapsed < 300 else '[FAIL] EXCEEDS 5-min limit!'})")
    print(f"  Top score : {top_candidates[0]['final_score'] if top_candidates else 'N/A'}")
    print(f"  Rank-100 score: {top_candidates[-1]['final_score'] if len(top_candidates) >= 100 else 'N/A'}")
    print("="*60 + "\n")

    if wall_elapsed >= 300:
        sys.exit(1)


if __name__ == "__main__":
    main()
