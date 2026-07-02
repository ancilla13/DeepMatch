"""
server.py — DeepMatch API Server
=================================
Serves ranked_candidates.json via a REST API and the frontend UI.
Supports live candidate/JD upload, pre-filtering, and dynamic ranking.

Usage:
    python server.py

Opens at: http://localhost:5000
"""

import json
import os
import time
from flask import Flask, jsonify, send_from_directory, send_file, request, Response
from werkzeug.utils import secure_filename
from docx import Document
import numpy as np

# Import core refactored functions
from prepare import run_prepare_pipeline, extract_grad_year
from rank import run_ranking_pipeline

app = Flask(__name__, static_folder="frontend", static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

RANKED_FILE = os.path.join(os.path.dirname(__file__), "ranked_candidates.json")

# In-memory store for the session (simple singleton for demo purposes)
session_store = {
    "candidates": None,      # List of raw candidate dicts
    "jd_text": None,         # Plain text of JD
    "model": None            # sentence-transformers model instance
}

def get_model():
    """Lazily load the model to speed up server start time, but keep it cached."""
    if session_store["model"] is None:
        print("  Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        session_store["model"] = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("  Model loaded successfully!")
    return session_store["model"]

@app.route("/")
def index():
    return send_file(os.path.join("frontend", "index.html"))

@app.route("/api/candidates")
def get_candidates():
    try:
        with open(RANKED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"status": "ok", "count": len(data), "candidates": data})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "ranked_candidates.json not found. Run rank.py first."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/stats")
def get_stats():
    try:
        with open(RANKED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        eligible = [c for c in data if not c.get("is_disqualified") and c.get("honeypot_flags", 0) < 2]
        disq = [c for c in data if c.get("is_disqualified")]
        honeypots = [c for c in data if c.get("honeypot_flags", 0) >= 2]
        scores = [c["final_score"] for c in eligible[:10]]

        return jsonify({
            "total": len(data),
            "eligible": len(eligible),
            "disqualified": len(disq),
            "honeypots": len(honeypots),
            "top10_avg": round(sum(scores) / len(scores), 4) if scores else 0,
            "top_score": eligible[0]["final_score"] if eligible else 0,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ── Live Upload & Custom Ranking Endpoints ────────────────────────────────────

@app.route("/api/upload/candidates", methods=["POST"])
def upload_candidates():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        filename = secure_filename(file.filename)
        content = file.read().decode('utf-8')
        
        candidates = []
        # Support JSON array
        try:
            data = json.loads(content)
            if isinstance(data, list):
                candidates = data
        except json.JSONDecodeError:
            # Fallback to JSONL (line-by-line)
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        if not candidates:
            return jsonify({"status": "error", "message": "No valid candidates parsed."}), 400

        # ENFORCE CHALLENGE SANDBOX LIMIT (Strictly <= 100)
        if len(candidates) > 100:
            return jsonify({
                "status": "error", 
                "message": f"Upload exceeds sandbox limit of 100 candidates (found {len(candidates)}). "
                           "Please upload <=100 candidates for demo."
            }), 400

        session_store["candidates"] = candidates
        return jsonify({
            "status": "ok", 
            "count": len(candidates), 
            "message": f"Successfully parsed {len(candidates)} candidates."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to parse candidates: {str(e)}"}), 500

@app.route("/api/upload/jd", methods=["POST"])
def upload_jd():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        filename = secure_filename(file.filename)
        if filename.endswith('.docx'):
            doc = Document(file)
            jd_text = "\n".join(para.text for para in doc.paragraphs)
        else:
            # Assume plain text
            jd_text = file.read().decode('utf-8')
        
        jd_text = jd_text.strip()
        if not jd_text:
            return jsonify({"status": "error", "message": "Job description text is empty."}), 400

        session_store["jd_text"] = jd_text
        return jsonify({
            "status": "ok", 
            "length": len(jd_text), 
            "preview": jd_text[:200] + "...",
            "message": "Successfully parsed Job Description."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to parse JD: {str(e)}"}), 500

@app.route("/api/run-ranking", methods=["GET", "POST"])
def run_ranking():
    candidates = session_store["candidates"]
    jd_text = session_store["jd_text"]

    if not candidates:
        return jsonify({"status": "error", "message": "No candidates uploaded yet."}), 400
    if not jd_text:
        return jsonify({"status": "error", "message": "No Job Description uploaded yet."}), 400

    if request.method == "POST":
        data = request.json or {}
    else:
        data = request.args or {}

    top_n = int(data.get("top_n", 10))
    yoe_filter = float(data.get("yoe_filter", 0))
    
    open_to_work_raw = data.get("open_to_work", False)
    if isinstance(open_to_work_raw, str):
        open_to_work = open_to_work_raw.lower() in ("true", "1", "yes")
    else:
        open_to_work = bool(open_to_work_raw)
    
    # Custom Server-Sent Events (SSE) connection to stream progress back to frontend
    def stream_progress():
        try:
            # Load model lazily
            yield "data: " + json.dumps({"step": "loading_model", "msg": "Initializing SentenceTransformer model..."}) + "\n\n"
            model = get_model()
            
            # Step 1: Loading dataset
            yield "data: " + json.dumps({"step": "loading_dataset", "msg": f"Parsing {len(candidates)} candidate profiles..."}) + "\n\n"
            time.sleep(0.5)

            # Step 2: Generating Embeddings
            yield "data: " + json.dumps({"step": "embeddings", "msg": "Generating semantic embeddings for JD and profiles..."}) + "\n\n"
            prep_results = run_prepare_pipeline(candidates, jd_text, model=model)
            
            # Step 3: Duplicate and scoring
            yield "data: " + json.dumps({"step": "scoring", "msg": "Running duplicate detection and scoring algorithms..."}) + "\n\n"
            top_candidates, sorted_all_results, duplicate_ids, disq_count, honeypot_count, skipped_count = run_ranking_pipeline(
                candidate_ids=prep_results["candidate_ids"],
                candidate_embeddings=prep_results["candidate_embeddings"],
                jd_embedding=prep_results["jd_embedding"],
                candidate_features=prep_results["candidate_features"],
                role_embeddings=prep_results["role_embeddings"],
                candidate_map={c["candidate_id"]: c for c in candidates},
                top_n=top_n,
                yoe_filter=yoe_filter,
                open_to_work_filter=open_to_work
            )

            # Enriched Results for dashboard UI
            json_results = []
            eligible_added = 0
            for r in sorted_all_results:
                is_disq = r["is_disqualified"]
                is_hp = r["honeypot_flags"] >= 2
                is_el = not is_disq and not is_hp
                
                if is_el:
                    if eligible_added >= top_n:
                        continue
                    eligible_added += 1
                
                cid = r["candidate_id"]
                raw_cand = next((c for c in candidates if c["candidate_id"] == cid), {})
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

            eligible = [c for c in json_results if not c.get("is_disqualified") and c.get("honeypot_flags", 0) < 2]
            disq = [c for c in json_results if c.get("is_disqualified")]
            honeypots = [c for c in json_results if c.get("honeypot_flags", 0) >= 2]
            top_scores = [c["final_score"] for c in eligible[:10]]

            stats = {
                "total": len(json_results),
                "eligible": len(eligible),
                "disqualified": len(disq),
                "honeypots": len(honeypots),
                "top10_avg": round(sum(top_scores) / len(top_scores), 4) if top_scores else 0,
                "top_score": eligible[0]["final_score"] if eligible else 0,
            }

            yield "data: " + json.dumps({
                "step": "done",
                "msg": "Ranking completed successfully!",
                "candidates": json_results,
                "stats": stats
            }) + "\n\n"
            
        except Exception as e:
            yield "data: " + json.dumps({"step": "error", "msg": f"Pipeline failure: {str(e)}"}) + "\n\n"

    return Response(stream_progress(), mimetype="text/event-stream")

if __name__ == "__main__":
    os.makedirs("frontend", exist_ok=True)
    print("\n DeepMatch Server starting...")
    print("   Dashboard: http://localhost:5000")
    print("   API:       http://localhost:5000/api/candidates\n")
    app.run(debug=True, port=5000)
