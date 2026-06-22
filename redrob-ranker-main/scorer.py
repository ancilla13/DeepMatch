"""
scorer.py
=========
Person 2 – Scoring Intelligence Layer
Redrob Intelligent Candidate Discovery & Ranking Challenge

This module provides all scoring and signal-processing functions consumed by
the ranking pipeline (rank.py).  It is intentionally side-effect-free:
- No print statements
- No file I/O
- No global mutable state
- No hardcoded paths
- Fully importable as a library

Python 3.10+ required.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable scoring weights
# ---------------------------------------------------------------------------

SEMANTIC_WEIGHT: float = 0.30
COVERAGE_WEIGHT: float = 0.20
TRAJECTORY_WEIGHT: float = 0.15
AUTHENTICITY_WEIGHT: float = 0.15
BEHAVIORAL_WEIGHT: float = 0.20
MAX_ANTI_PATTERN_PENALTY: float = 0.25

# ---------------------------------------------------------------------------
# JD must-have keyword categories
# ---------------------------------------------------------------------------

JD_MUST_HAVES: dict[str, list[str]] = {
    "embeddings_retrieval": [
        "embedding", "embeddings", "dense retrieval", "bi-encoder",
        "semantic search", "faiss", "ann", "approximate nearest neighbour",
        "approximate nearest neighbor", "sentence-transformers", "sentence transformers",
        "retrieval augmented", "rag", "retrieval-augmented generation",
        "vector search", "knn", "nearest neighbor",
    ],
    "vector_db_hybrid": [
        "pinecone", "weaviate", "qdrant", "milvus", "chroma", "chromadb",
        "pgvector", "opensearch", "elasticsearch", "hybrid search",
        "sparse", "bm25", "keyword search", "vector database", "vector db",
        "vector store",
    ],
    "strong_python": [
        "python", "fastapi", "flask", "django", "asyncio", "pydantic",
        "numpy", "pandas", "scipy", "pytorch", "tensorflow", "jax",
        "scikit-learn", "sklearn",
    ],
    "eval_frameworks": [
        "ragas", "trulens", "mlflow", "wandb", "weights & biases",
        "evaluation", "benchmarking", "metrics", "precision", "recall",
        "ndcg", "mrr", "map@k", "a/b testing", "ab testing",
        "unit test", "pytest", "test coverage",
    ],
}

# ---------------------------------------------------------------------------
# Proficiency level mapping (canonical names → numeric depth)
# ---------------------------------------------------------------------------

PROFICIENCY_MAP: dict[str, int] = {
    "beginner": 1,
    "elementary": 1,
    "basic": 1,
    "novice": 1,
    "intermediate": 2,
    "moderate": 2,
    "professional": 2,
    "advanced": 3,
    "experienced": 3,
    "proficient": 3,
    "expert": 4,
    "master": 4,
    "principal": 4,
}

# Minimum months of practice expected per proficiency level
# (used as a sanity check in skill_authenticity)
PROFICIENCY_MIN_MONTHS: dict[int, int] = {
    1: 0,    # beginner  – no minimum
    2: 6,    # intermediate – ~6 months
    3: 18,   # advanced – ~1.5 years
    4: 36,   # expert   – ~3 years
}

# Recency weights for trajectory scoring (newest-first)
TRAJECTORY_WEIGHTS: list[float] = [1.0, 0.8, 0.5, 0.3, 0.2]

# Inactivity threshold in days before a penalty is applied
INACTIVITY_PENALTY_DAYS: int = 180

# Title escalation ladder (ordered lowest → highest)
TITLE_ESCALATION_LADDERS: list[list[str]] = [
    # Engineering ladder
    ["engineer", "senior engineer", "staff engineer", "principal engineer",
     "distinguished engineer", "fellow"],
    # Manager / leadership ladder
    ["associate", "analyst", "senior analyst", "lead", "manager",
     "senior manager", "director", "vp", "vice president", "cto", "ceo"],
    # ML-specific ladder
    ["ml engineer", "senior ml engineer", "staff ml engineer",
     "principal ml engineer", "ml architect"],
    # Data science ladder
    ["data scientist", "senior data scientist", "lead data scientist",
     "principal data scientist", "staff data scientist"],
    # Research ladder
    ["research engineer", "senior research engineer",
     "staff research engineer", "principal research engineer"],
    # Generic seniority keywords in rank order
    ["junior", "mid", "senior", "staff", "principal", "distinguished", "fellow"],
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Return *value* as a float, falling back to *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def _normalize_embedding(vec: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of *vec*, or a zero vector if it is near-zero."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return np.zeros_like(vec)
    return vec / norm


def _resolve_proficiency(raw: Any) -> int:
    """Map a proficiency string/int to a numeric depth 1–4.

    Accepts string labels (case-insensitive) or integers in [1, 4].
    Returns 1 (beginner) for unrecognised values.
    """
    if isinstance(raw, int):
        return max(1, min(4, raw))
    if isinstance(raw, float):
        return max(1, min(4, int(raw)))
    if isinstance(raw, str):
        return PROFICIENCY_MAP.get(raw.strip().lower(), 1)
    return 1


# ---------------------------------------------------------------------------
# Function 1 – Semantic score
# ---------------------------------------------------------------------------

def semantic_score(
    candidate_embedding: np.ndarray,
    jd_embedding: np.ndarray,
) -> float:
    """Compute the cosine similarity between a candidate and JD embedding.

    Parameters
    ----------
    candidate_embedding:
        1-D or 2-D numpy array representing the candidate's text embedding.
    jd_embedding:
        1-D or 2-D numpy array representing the job-description embedding.

    Returns
    -------
    float
        Cosine similarity in [0, 1].  Returns 0.0 if either vector is
        near-zero (avoids divide-by-zero artefacts).
    """
    # Ensure 1-D
    cand = np.asarray(candidate_embedding, dtype=np.float32).flatten()
    jd = np.asarray(jd_embedding, dtype=np.float32).flatten()

    # Guard against zero vectors
    if np.linalg.norm(cand) < 1e-10 or np.linalg.norm(jd) < 1e-10:
        return 0.0

    # sklearn expects 2-D arrays
    sim = cosine_similarity(cand.reshape(1, -1), jd.reshape(1, -1))[0][0]

    # Cosine similarity is in [-1, 1]; clamp to [0, 1] for scoring purposes
    return float(_clamp(float(sim), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Function 2 – Disqualification check
# ---------------------------------------------------------------------------

def is_disqualified(
    candidate: dict[str, Any],
    features: dict[str, Any],
) -> bool:
    """Return True if the candidate triggers any hard-disqualification flag.

    Disqualifying conditions (any one is sufficient):
    - ``pure_research_no_production`` – research-only background, no production work.
    - ``ai_only_langchain_under_12mo`` – exclusively LangChain/agent demos < 12 months.
    - ``no_hands_on_code_18mo`` – no hands-on coding in the last 18 months.
    - ``consulting_only_career`` – career limited to advisory / consulting roles.
    - ``cv_speech_robotics_no_nlp`` – CV/Speech/Robotics background with no NLP.

    Parameters
    ----------
    candidate:
        Raw candidate dictionary (not used directly; reserved for extensions).
    features:
        Extracted features dictionary from ``prepare.py``.

    Returns
    -------
    bool
    """
    _flags = [
        "pure_research_no_production",
        "ai_only_langchain_under_12mo",
        "no_hands_on_code_18mo",
        "consulting_only_career",
        "cv_speech_robotics_no_nlp",
    ]

    missing: list[str] = []
    for flag in _flags:
        if flag not in features:
            missing.append(flag)
        elif features[flag]:
            return True

    if missing:
        logger.debug(
            "Disqualification flags absent from features (defaulting to "
            "False): %s",
            ", ".join(missing),
        )

    return False


def disqualification_detail(
    candidate: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    """Return a detailed breakdown of the disqualification check.

    Unlike ``is_disqualified``, this returns a dictionary with:

    - ``is_disqualified`` (bool): overall result.
    - ``triggered_flag`` (str | None): the first flag that triggered, or None.
    - ``flags_checked`` (dict[str, bool | None]): per-flag status.
      ``None`` means the flag was absent from *features*.

    Parameters
    ----------
    candidate:
        Raw candidate dictionary.
    features:
        Extracted features dictionary.

    Returns
    -------
    dict[str, Any]
    """
    _flags = [
        "pure_research_no_production",
        "ai_only_langchain_under_12mo",
        "no_hands_on_code_18mo",
        "consulting_only_career",
        "cv_speech_robotics_no_nlp",
    ]

    flags_checked: dict[str, bool | None] = {}
    triggered_flag: str | None = None

    for flag in _flags:
        if flag not in features:
            flags_checked[flag] = None
        else:
            value = bool(features[flag])
            flags_checked[flag] = value
            if value and triggered_flag is None:
                triggered_flag = flag

    return {
        "is_disqualified": triggered_flag is not None,
        "triggered_flag": triggered_flag,
        "flags_checked": flags_checked,
    }


# ---------------------------------------------------------------------------
# Function 3 – Honeypot score
# ---------------------------------------------------------------------------

def honeypot_score(
    candidate: dict[str, Any],
    features: dict[str, Any],
) -> int:
    """Count profile-integrity red flags (higher = more suspicious).

    Each of the following contributes +1 to the flag count:

    1. ``has_overlapping_dates`` – overlapping employment periods detected.
    2. ``started_before_graduating`` – work started suspiciously before graduation.
    3. ``duration_mismatch_count`` > 0 – declared vs. actual duration mismatch.
    4. Any skill claimed at "expert" level with ≤ 2 months practice duration.

    The pipeline treats ``flags >= 2`` as a honeypot candidate.

    Parameters
    ----------
    candidate:
        Raw candidate dictionary; ``skills`` list is inspected for (4).
    features:
        Extracted features dictionary.

    Returns
    -------
    int
        Total flag count (0 = clean profile).
    """
    flags = 0

    if features.get("has_overlapping_dates", False):
        flags += 1

    if features.get("started_before_graduating", False):
        flags += 1

    if int(features.get("duration_mismatch_count", 0)) > 0:
        flags += 1

    # Check for expert-level skills with negligible practice time
    skills: list[dict[str, Any]] = candidate.get("skills", [])
    for skill in skills:
        proficiency = _resolve_proficiency(skill.get("proficiency_level", 1))
        duration_months = _safe_float(skill.get("duration_months", 0))

        # Expert claimed with ≤ 2 months is implausible
        if proficiency == 4 and duration_months <= 2:
            flags += 1
            break  # One flag per candidate from this check is sufficient

    return flags


# ---------------------------------------------------------------------------
# Function 4 – JD requirement coverage
# ---------------------------------------------------------------------------

def jd_requirement_coverage(
    text_blob: str | None,
    skills: list[dict[str, Any]],
) -> float:
    """Estimate what fraction of JD must-have categories the candidate covers.

    Matching is case-insensitive and applied to:
    - ``text_blob``    – candidate's full text (headline + summary + roles).
    - ``skills``       – list of skill dicts, each with at least a ``name`` key.

    Parameters
    ----------
    text_blob:
        Concatenated free-text profile of the candidate.
    skills:
        List of skill dictionaries from the candidate schema.

    Returns
    -------
    float
        ``covered_categories / 4`` in [0.0, 1.0].
    """
    # Build a unified lowercase search corpus
    blob_lower = text_blob.lower() if text_blob else ""
    skill_names_lower = " ".join(
        s.get("name", "").lower() for s in skills if isinstance(s, dict)
    )
    corpus = blob_lower + " " + skill_names_lower

    covered = 0

    for category, keywords in JD_MUST_HAVES.items():
        for kw in keywords:
            if kw.lower() in corpus:
                covered += 1
                break  # Category is covered; move on

    return covered / len(JD_MUST_HAVES)  # Always 4 categories


# ---------------------------------------------------------------------------
# Function 5 – Skill authenticity
# ---------------------------------------------------------------------------

def skill_authenticity(
    skills: list[dict[str, Any]],
    assessment_scores: dict[str, float] | None = None,
) -> float:
    """Score the credibility of a candidate's skill claims.

    Each skill contributes a per-skill authenticity score based on:

    1. **Proficiency–Duration coherence** – whether the claimed proficiency
       is plausible given the reported practice duration.
    2. **Assessment scores** – verified test results boost the raw score.
    3. **Endorsement factor** – social proof from peers.

    The final score is a weighted average across all skills.

    Per-skill formula
    ~~~~~~~~~~~~~~~~~
    ::

        # Duration coherence: how many times over the minimum has the
        # candidate met the required months for their claimed level?
        coherence = min(duration_months / min_months, 1.5) / 1.5
                    (= 1.0 if min_months == 0)

        # Assessment boost: 0.0 if no assessment; normalised [0, 1]
        assessment_boost = assessment_score / 100  (clamped to [0, 1])

        # Endorsement factor: log-saturated endorsement signal
        endorsement_factor = min(log(1 + endorsements) / log(51), 1.0)
                            (caps out at ~50 endorsements)

        # Weights: coherence 50%, assessment 30%, endorsements 20%
        skill_score = (0.50 * coherence
                       + 0.30 * assessment_boost
                       + 0.20 * endorsement_factor)

    The final score is the arithmetic mean of all per-skill scores,
    clamped to [0.0, 1.0].

    Parameters
    ----------
    skills:
        List of skill dicts.  Each dict may contain:
        - ``name`` (str)
        - ``proficiency_level`` (str | int)
        - ``duration_months`` (int | float)
        - ``endorsements`` (int)
    assessment_scores:
        Optional mapping of skill name (lowercase) → score in [0, 100].

    Returns
    -------
    float
        Authenticity score in [0.0, 1.0].  Returns 0.5 (neutral) for
        an empty skill list.
    """
    if not skills:
        return 0.5  # Neutral: no evidence for or against

    if assessment_scores is None:
        assessment_scores = {}

    per_skill_scores: list[float] = []

    for skill in skills:
        if not isinstance(skill, dict):
            continue

        proficiency = _resolve_proficiency(skill.get("proficiency_level", 1))
        duration_months = _safe_float(skill.get("duration_months", 0))
        endorsements = int(_safe_float(skill.get("endorsements", 0)))
        skill_name = (skill.get("name") or "").lower().strip()

        # --- 1. Proficiency–duration coherence ---
        min_months = PROFICIENCY_MIN_MONTHS.get(proficiency, 0)
        if min_months == 0:
            # Beginner or unrecognised level – no minimum required
            coherence = 1.0 if duration_months >= 0 else 0.0
        else:
            ratio = duration_months / min_months
            # Allow up to 1.5× the minimum for full coherence credit
            coherence = _clamp(ratio / 1.5, 0.0, 1.0)

        # --- 2. Assessment boost ---
        raw_assessment = assessment_scores.get(skill_name, None)
        if raw_assessment is None and skill_name:
            # Check for partial name match (skip if skill_name is empty)
            raw_assessment = next(
                (v for k, v in assessment_scores.items() if k in skill_name or skill_name in k),
                None,
            )
        assessment_boost = (
            _clamp(_safe_float(raw_assessment) / 100.0)
            if raw_assessment is not None
            else 0.0
        )

        # --- 3. Endorsement factor (log-saturated) ---
        endorsement_factor = _clamp(
            np.log1p(endorsements) / np.log1p(50), 0.0, 1.0
        )

        # --- Weighted combination ---
        skill_score = (
            0.50 * coherence
            + 0.30 * assessment_boost
            + 0.20 * endorsement_factor
        )
        per_skill_scores.append(_clamp(skill_score))

    if not per_skill_scores:
        return 0.5

    return float(_clamp(float(np.mean(per_skill_scores))))


# ---------------------------------------------------------------------------
# Function 6 – Trajectory score
# ---------------------------------------------------------------------------

def trajectory_score(
    role_embeddings_sorted_newest_first: list[list[float]] | list[np.ndarray],
    jd_embedding: np.ndarray,
) -> float:
    """Score how closely a candidate's career trajectory aligns with the JD.

    Recency weighting ensures recent roles count more than older ones.
    Weights (newest → oldest): [1.0, 0.8, 0.5, 0.3, 0.2].

    Parameters
    ----------
    role_embeddings_sorted_newest_first:
        List of role embeddings ordered from most recent to oldest.
    jd_embedding:
        Job description embedding.

    Returns
    -------
    float
        Weighted average cosine similarity in [0.0, 1.0].
        Returns 0.0 if the list is empty or JD embedding is zero.
    """
    if not role_embeddings_sorted_newest_first:
        return 0.0

    jd = np.asarray(jd_embedding, dtype=np.float32).flatten()
    if np.linalg.norm(jd) < 1e-10:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    for idx, role_emb in enumerate(role_embeddings_sorted_newest_first):
        weight = TRAJECTORY_WEIGHTS[idx] if idx < len(TRAJECTORY_WEIGHTS) else TRAJECTORY_WEIGHTS[-1]

        role = np.asarray(role_emb, dtype=np.float32).flatten()
        if np.linalg.norm(role) < 1e-10:
            # Skip degenerate embeddings
            continue

        sim = cosine_similarity(role.reshape(1, -1), jd.reshape(1, -1))[0][0]
        sim_clamped = float(_clamp(float(sim), 0.0, 1.0))

        weighted_sum += weight * sim_clamped
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    return float(_clamp(weighted_sum / total_weight))


# ---------------------------------------------------------------------------
# Function 7 – Behavioral score
# ---------------------------------------------------------------------------

def behavioral_score(signals: dict[str, Any]) -> float:
    """Convert Redrob behavioral signals into a single score in [0, 1].

    Signals used
    ~~~~~~~~~~~~
    - ``open_to_work_flag`` (bool): +bonus for active job seekers.
    - ``last_active_date`` (str, ISO format): penalty for >180 days inactive.
    - ``recruiter_response_rate`` (float, 0–1): responsiveness to outreach.
    - ``interview_completion_rate`` (float, 0–1): follow-through on interviews.
    - ``offer_acceptance_rate`` (float, 0–1): closing intent.
    - ``github_activity_score`` (float, 0–1 or 0–100): open-source engagement.
    - ``verified_email`` (bool): trust signal.
    - ``verified_phone`` (bool): trust signal.
    - ``linkedin_connected`` (bool): profile authenticity.

    Scoring breakdown
    ~~~~~~~~~~~~~~~~~
    ::

        engagement  = mean(recruiter_response_rate,
                           interview_completion_rate,
                           offer_acceptance_rate)          weight 0.50
        trust       = mean(verified_email, verified_phone,
                           linkedin_connected)             weight 0.25
        activity    = github_activity_score (normalised)   weight 0.25

    Modifiers:
    - open_to_work_flag=True → +0.05 bonus (capped at 1.0)
    - inactive > 180 days   → −0.10 penalty

    Parameters
    ----------
    signals:
        Redrob signals dictionary (may contain any subset of the above keys).

    Returns
    -------
    float
        Behavioral score in [0.0, 1.0].
    """
    if not signals:
        return 0.0

    # --- Engagement sub-score ---
    recruiter_response = _safe_float(signals.get("recruiter_response_rate"), -1)
    interview_completion = _safe_float(signals.get("interview_completion_rate"), -1)
    offer_acceptance = _safe_float(signals.get("offer_acceptance_rate"), -1)

    engagement_values = [v for v in [recruiter_response, interview_completion, offer_acceptance] if v >= 0]
    engagement = float(np.mean(engagement_values)) if engagement_values else 0.5  # neutral if missing

    # --- Trust sub-score ---
    verified_email = bool(signals.get("verified_email", False))
    verified_phone = bool(signals.get("verified_phone", False))
    linkedin_connected = bool(signals.get("linkedin_connected", False))
    trust = (int(verified_email) + int(verified_phone) + int(linkedin_connected)) / 3.0

    # --- Activity sub-score (GitHub) ---
    raw_github = _safe_float(signals.get("github_activity_score"), -1)
    if raw_github < 0:
        activity = 0.5  # neutral if missing
    elif raw_github > 1.0:
        # Treat as 0–100 scale; normalise to 0–1
        activity = _clamp(raw_github / 100.0)
    else:
        activity = _clamp(raw_github)

    # --- Weighted combination ---
    score = 0.50 * _clamp(engagement) + 0.25 * trust + 0.25 * activity

    # --- Modifiers ---
    if signals.get("open_to_work_flag", False):
        score += 0.05

    # Inactivity penalty
    last_active_raw = signals.get("last_active_date")
    if last_active_raw:
        try:
            # Accept ISO 8601 strings with or without timezone
            last_active_str = str(last_active_raw).rstrip("Z")
            last_active_dt = datetime.fromisoformat(last_active_str)
            # Make both timezone-aware or both naive for comparison
            if last_active_dt.tzinfo:
                now = datetime.now(timezone.utc)
            else:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
            days_inactive = (now - last_active_dt).days
            if days_inactive > INACTIVITY_PENALTY_DAYS:
                score -= 0.10
        except (ValueError, TypeError, AttributeError):
            pass  # Unparseable date – skip modifier

    return float(_clamp(score))


# ---------------------------------------------------------------------------
# Function 8 – Title escalation detection
# ---------------------------------------------------------------------------

def title_shows_escalation(career_history: list[dict[str, Any]]) -> bool:
    """Detect genuine upward title progression in a candidate's career.

    A career shows escalation if at least two consecutive roles (ordered
    chronologically) progress **forward** along a recognised seniority ladder.

    Parameters
    ----------
    career_history:
        List of role dicts, each with at least a ``title`` (str) and
        optionally ``start_date`` (str, "YYYY-MM").

    Returns
    -------
    bool
    """
    if not career_history or len(career_history) < 2:
        return False

    # Sort chronologically (oldest first) for progression analysis
    def _parse_start(role: dict) -> datetime:
        raw = role.get("start_date", "")
        try:
            return datetime.strptime(str(raw)[:7], "%Y-%m")
        except (ValueError, TypeError):
            return datetime.min

    sorted_history = sorted(career_history, key=_parse_start)
    titles = [r.get("title", "").lower().strip() for r in sorted_history]

    def _ladder_rank(title: str) -> dict[str, int]:
        """Return {ladder_name: rank} for a title across all ladders."""
        ranks: dict[str, int] = {}
        for ladder_name, ladder in enumerate(TITLE_ESCALATION_LADDERS):
            for rank, level in enumerate(ladder):
                if level in title:
                    # Keep the highest rank found on this ladder
                    if ladder_name not in ranks or rank > ranks[ladder_name]:
                        ranks[ladder_name] = rank
        return ranks

    # Slide a window over consecutive pairs and detect forward progression
    for i in range(len(titles) - 1):
        current_ranks = _ladder_rank(titles[i])
        next_ranks = _ladder_rank(titles[i + 1])

        # Check if any common ladder shows upward progression
        for ladder_name, current_rank in current_ranks.items():
            next_rank = next_ranks.get(ladder_name)
            if next_rank is not None and next_rank > current_rank:
                return True

    return False


# ---------------------------------------------------------------------------
# Function 9 – Anti-pattern penalty
# ---------------------------------------------------------------------------

def anti_pattern_penalty(
    career_history: list[dict[str, Any]] | None,
    text_blob: str | None,
) -> float:
    """Calculate a penalty for observable anti-patterns in a candidate's profile.

    Anti-patterns detected
    ~~~~~~~~~~~~~~~~~~~~~~

    1. **Title Chaser** (−0.15)
       Triggered when ALL of the following hold:
       - 3 or more jobs exist.
       - The majority of those jobs lasted < 18 months.
       - The titles show an escalation pattern (``title_shows_escalation``).

    2. **Framework Enthusiast** (−0.10)
       Triggered when the text blob mentions 2+ "toy-project" keywords AND
       contains no mention of production work.

    Parameters
    ----------
    career_history:
        List of role dicts.
    text_blob:
        Candidate's full concatenated free-text profile.

    Returns
    -------
    float
        Total penalty in [0.0, MAX_ANTI_PATTERN_PENALTY].
    """
    penalty = 0.0
    blob_lower = text_blob.lower() if text_blob else ""

    # --- Anti-pattern 1: Title chaser ---
    if career_history:
        TITLE_CHASER_MIN_JOBS = 3
        TITLE_CHASER_MAX_DURATION_MONTHS = 18

        if len(career_history) >= TITLE_CHASER_MIN_JOBS:
            short_tenures = sum(
                1 for role in career_history
                if _safe_float(role.get("duration_months", 99)) < TITLE_CHASER_MAX_DURATION_MONTHS
            )
            majority_short = short_tenures >= (len(career_history) / 2)

            if majority_short and title_shows_escalation(career_history):
                TITLE_CHASER_PENALTY = 0.15
                penalty += TITLE_CHASER_PENALTY

    # --- Anti-pattern 2: Framework enthusiast ---
    FRAMEWORK_KEYWORDS = [
        "tutorial",
        "demo project",
        "toy project",
        "chatbot",
        "poc",
    ]
    PRODUCTION_KEYWORDS = [
        "production",
        "prod",
        "deployed",
        "serving",
        "real-world",
        "live system",
        "scale",
        "customers",
        "users",
    ]
    FRAMEWORK_ENTHUSIASM_PENALTY = 0.10
    FRAMEWORK_ENTHUSIASM_THRESHOLD = 2

    keyword_hits = sum(1 for kw in FRAMEWORK_KEYWORDS if kw in blob_lower)
    has_production = any(kw in blob_lower for kw in PRODUCTION_KEYWORDS)

    if keyword_hits >= FRAMEWORK_ENTHUSIASM_THRESHOLD and not has_production:
        penalty += FRAMEWORK_ENTHUSIASM_PENALTY

    return float(min(penalty, MAX_ANTI_PATTERN_PENALTY))


# ---------------------------------------------------------------------------
# Function 10 – Duplicate flagging
# ---------------------------------------------------------------------------

def flag_duplicates(
    embeddings: np.ndarray,
    candidate_ids: list[str],
    behavioral_scores: list[float] | None = None,
    threshold: float = 0.97,
    chunk_size: int = 500,
) -> set[str]:
    """Identify near-duplicate candidate profiles via embedding similarity.

    Algorithm
    ~~~~~~~~~
    1. L2-normalise all embeddings (unit-sphere projection).
    2. Compute the full N×N cosine similarity matrix efficiently using a
       single matrix multiplication (O(N²) space; appropriate for typical
       candidate pool sizes).
    3. For each pair with similarity ≥ *threshold*, mark the candidate with
       the lower behavioral score for removal.  If behavioral scores are not
       available, keep the candidate that appears first in the list.

    Parameters
    ----------
    embeddings:
        N×D matrix of candidate embeddings.
    candidate_ids:
        List of N candidate ID strings (must align with *embeddings*).
    behavioral_scores:
        Optional list of N float scores in [0, 1].  Higher is better.
    threshold:
        Cosine similarity threshold above which two candidates are
        considered duplicates.  Default 0.97.

    Returns
    -------
    set[str]
        Set of candidate IDs to remove from the ranking pool.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = embeddings.shape[0]

    if n == 0 or len(candidate_ids) != n:
        return set()

    if n > 50_000:
        logger.warning(
            "flag_duplicates: N=%d exceeds 50,000. "
            "Consider FAISS IndexFlatIP for production-scale dedup.",
            n,
        )

    # Normalise embeddings row-wise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms < 1e-10, 1.0, norms)
    normed = embeddings / norms  # shape (N, D)

    # Cosine similarity matrix (N×N) via dot product on unit vectors
    to_remove: set[str] = set()

    # Process in chunks to avoid O(N^2) memory allocation.
    # Peak memory per chunk: O(chunk_size * (N - offset)).
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        chunk = normed[i_start:i_end]

        # Compute similarities only with candidates at index >= i_start
        # to avoid redundant pair checks and reduce peak memory.
        tail = normed[i_start:]
        sim_block = chunk @ tail.T  # shape (chunk_len, N - i_start)

        for local_i in range(i_end - i_start):
            global_i = i_start + local_i
            if candidate_ids[global_i] in to_remove:
                continue

            # Only check j > global_i (row slice after self)
            row_after_self = sim_block[local_i, local_i + 1:]
            above_indices = np.where(row_after_self >= threshold)[0]

            for offset in above_indices:
                global_j = global_i + 1 + offset
                if candidate_ids[global_j] in to_remove:
                    continue

                if behavioral_scores is not None:
                    score_i = _safe_float(
                        behavioral_scores[global_i], 0.0,
                    )
                    score_j = _safe_float(
                        behavioral_scores[global_j], 0.0,
                    )
                    loser = (
                        candidate_ids[global_j]
                        if score_i >= score_j
                        else candidate_ids[global_i]
                    )
                else:
                    loser = candidate_ids[global_j]

                to_remove.add(loser)

    return to_remove


# ---------------------------------------------------------------------------
# Function 11 – Final composite score
# ---------------------------------------------------------------------------

def final_score(
    semantic_fit: float,
    coverage: float,
    trajectory: float,
    authenticity: float,
    behavioral: float,
    anti_pattern_penalty_value: float,
    is_disq: bool,
    honeypot_flags: int,
) -> float:
    """Compute the final composite candidate score.

    Hard zeroes
    ~~~~~~~~~~~
    - ``is_disq=True``        → 0.0 (disqualified)
    - ``honeypot_flags >= 2`` → 0.0 (suspected fake/inflated profile)

    Otherwise
    ~~~~~~~~~
    ::

        raw = (SEMANTIC_WEIGHT    × semantic_fit
               + COVERAGE_WEIGHT   × coverage
               + TRAJECTORY_WEIGHT × trajectory
               + AUTHENTICITY_WEIGHT × authenticity
               + BEHAVIORAL_WEIGHT × behavioral)

        score = clamp(raw − anti_pattern_penalty, 0, 1)

    Parameters
    ----------
    semantic_fit:
        Output of ``semantic_score`` (0–1).
    coverage:
        Output of ``jd_requirement_coverage`` (0–1).
    trajectory:
        Output of ``trajectory_score`` (0–1).
    authenticity:
        Output of ``skill_authenticity`` (0–1).
    behavioral:
        Output of ``behavioral_score`` (0–1).
    anti_pattern_penalty_value:
        Output of ``anti_pattern_penalty`` (0–MAX_ANTI_PATTERN_PENALTY).
    is_disq:
        Output of ``is_disqualified``.
    honeypot_flags:
        Output of ``honeypot_score``.

    Returns
    -------
    float
        Final score rounded to 4 decimal places in [0.0, 1.0].
    """
    if is_disq:
        return 0.0

    HONEYPOT_THRESHOLD = 2
    if honeypot_flags >= HONEYPOT_THRESHOLD:
        return 0.0

    raw = (
        SEMANTIC_WEIGHT * _clamp(semantic_fit)
        + COVERAGE_WEIGHT * _clamp(coverage)
        + TRAJECTORY_WEIGHT * _clamp(trajectory)
        + AUTHENTICITY_WEIGHT * _clamp(authenticity)
        + BEHAVIORAL_WEIGHT * _clamp(behavioral)
    )

    adjusted = raw - _clamp(anti_pattern_penalty_value, 0.0, MAX_ANTI_PATTERN_PENALTY)

    return round(_clamp(adjusted), 4)


# ---------------------------------------------------------------------------
# Function 12 – Generate recruiter reasoning
# ---------------------------------------------------------------------------

def generate_reasoning(
    candidate: dict[str, Any],
    scores: dict[str, float],
) -> str:
    """Generate a factual, evidence-based recruiter-grade justification.

    The output is deterministic (no LLMs, no randomness) and constructed
    exclusively from verifiable candidate fields and computed scores.

    Parameters
    ----------
    candidate:
        Raw candidate dictionary with at least a ``profile`` key.
    scores:
        Dictionary of computed scores, e.g.::

            {
                "semantic_fit": 0.82,
                "coverage": 0.75,
                "trajectory": 0.68,
                "authenticity": 0.91,
                "behavioral": 0.60,
                "final": 0.7314,
            }

    Returns
    -------
    str
        1–2 sentence evidence-based justification.
    """
    profile: dict[str, Any] = candidate.get("profile", {})

    # Support both nested profile schema (raw candidate) and
    # flattened feature schema (from candidate_features.json)
    title = (
        profile.get("current_title", "")
        or candidate.get("current_title", "")
        or profile.get("headline", "")
        or "N/A"
    )
    company = (
        profile.get("current_company", "")
        or candidate.get("current_company", "")
        or "N/A"
    )
    # Explicit None check for YOE since 0 is a valid value
    yoe_raw = profile.get("years_of_experience")
    if yoe_raw is None:
        yoe_raw = candidate.get("yoe")
    yoe = _safe_float(yoe_raw if yoe_raw is not None else 0)

    # Top skills (up to 3 named skills for brevity)
    skills: list[dict[str, Any]] = candidate.get("skills", [])
    top_skill_names = [
        s["name"] for s in skills[:3] if isinstance(s, dict) and s.get("name")
    ]

    # Redrob behavioral signals
    signals: dict[str, Any] = candidate.get("redrob_signals", {})
    verified_email = bool(signals.get("verified_email", False))
    recruiter_response_rate = _safe_float(signals.get("recruiter_response_rate", -1))
    open_to_work = bool(signals.get("open_to_work_flag", False))

    # --- Sentence 1: Experience & role ---
    yoe_str = f"{yoe:.1f}" if yoe > 0 else "an unspecified number of"
    sentence1_parts = [
        f"Candidate brings {yoe_str} years of experience"
    ]
    if title != "N/A":
        sentence1_parts[0] += f" and currently serves as {title}"
    if company != "N/A":
        sentence1_parts[0] += f" at {company}"
    sentence1 = sentence1_parts[0] + "."

    # --- Sentence 2: Strengths and signal summary ---
    strength_items: list[str] = []

    # Semantic / coverage signal
    semantic_fit = _safe_float(scores.get("semantic_fit", -1))
    if semantic_fit >= 0.75:
        strength_items.append("strong semantic alignment to the JD")
    elif semantic_fit >= 0.50:
        strength_items.append("moderate semantic fit")

    coverage = _safe_float(scores.get("coverage", -1))
    if coverage >= 0.75:
        strength_items.append("broad coverage of JD must-haves")
    elif coverage >= 0.50:
        strength_items.append("partial coverage of required skills")

    # Named skills
    if top_skill_names:
        skills_str = ", ".join(top_skill_names)
        strength_items.append(f"demonstrated expertise in {skills_str}")

    # Trajectory
    trajectory = _safe_float(scores.get("trajectory", -1))
    if trajectory >= 0.70:
        strength_items.append("a progressively relevant career trajectory")

    # Authenticity
    authenticity = _safe_float(scores.get("authenticity", -1))
    if authenticity >= 0.80:
        strength_items.append("highly credible skill claims")
    elif authenticity < 0.40 and authenticity >= 0:
        strength_items.append("some unverified skill claims")

    # Behavioral / recruiter signals
    if recruiter_response_rate >= 0.7:
        strength_items.append("high recruiter responsiveness")
    if open_to_work:
        strength_items.append("actively open to new opportunities")
    if verified_email:
        strength_items.append("a verified contact profile")

    # Final score verdict
    final = _safe_float(scores.get("final", -1))
    if final >= 0.70:
        verdict = "supporting a strong match for the role"
    elif final >= 0.50:
        verdict = "indicating a moderate fit for the role"
    elif final >= 0:
        verdict = "suggesting a below-average fit for this role"
    else:
        verdict = ""

    if strength_items:
        strengths_str = "; ".join(strength_items)
        sentence2 = f"Their profile demonstrates {strengths_str}"
        if verdict:
            sentence2 += f", {verdict}."
        else:
            sentence2 += "."
    else:
        sentence2 = f"Insufficient signal data for a detailed assessment." if not verdict else f"Profile data {verdict}."

    return f"{sentence1} {sentence2}".strip()


# ---------------------------------------------------------------------------
# Self-test block (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list[tuple[str, bool]] = []

    def _check(name: str, condition: bool) -> None:
        results.append((name, condition))
        status = PASS if condition else FAIL
        sys.stdout.write(f"  [{status}] {name}\n")

    sys.stdout.write("\n=== scorer.py self-test ===\n\n")

    # ---- Mock data --------------------------------------------------------
    RNG = np.random.default_rng(42)
    DIM = 384

    mock_jd_emb = RNG.random(DIM).astype(np.float32)
    mock_cand_emb = mock_jd_emb * 0.95 + RNG.random(DIM).astype(np.float32) * 0.05
    mock_dissimilar_emb = -mock_jd_emb  # Opposite direction

    mock_role_embs = [
        (mock_jd_emb * 0.9 + RNG.random(DIM) * 0.1).tolist(),
        (mock_jd_emb * 0.7 + RNG.random(DIM) * 0.3).tolist(),
        (mock_jd_emb * 0.4 + RNG.random(DIM) * 0.6).tolist(),
    ]

    mock_candidate = {
        "candidate_id": "MOCK_001",
        "profile": {
            "current_title": "Senior ML Engineer",
            "current_company": "TechCorp AI",
            "headline": "Senior ML Engineer specialising in retrieval systems",
            "summary": "Built production RAG pipelines with FAISS and Python.",
            "years_of_experience": 7.2,
        },
        "career_history": [
            {
                "title": "Junior Engineer",
                "company": "StartupA",
                "start_date": "2017-06",
                "end_date": "2019-06",
                "duration_months": 24,
                "description": "Built Python ETL pipelines.",
                "is_current": False,
            },
            {
                "title": "Senior Engineer",
                "company": "ScaleupB",
                "start_date": "2019-08",
                "end_date": "2022-08",
                "duration_months": 36,
                "description": "Deployed production vector search systems at scale.",
                "is_current": False,
            },
            {
                "title": "Senior ML Engineer",
                "company": "TechCorp AI",
                "start_date": "2022-09",
                "end_date": None,
                "duration_months": 22,
                "description": "Architected RAG pipelines with FAISS and embeddings.",
                "is_current": True,
            },
        ],
        "education": [
            {"degree": "B.Tech", "field": "Computer Science", "end_year": 2017}
        ],
        "skills": [
            {"name": "Python", "proficiency_level": "expert", "duration_months": 84, "endorsements": 30},
            {"name": "FAISS", "proficiency_level": "advanced", "duration_months": 36, "endorsements": 12},
            {"name": "Embeddings", "proficiency_level": "advanced", "duration_months": 30, "endorsements": 8},
            {"name": "LangChain", "proficiency_level": "intermediate", "duration_months": 12, "endorsements": 5},
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "last_active_date": "2026-06-01",
            "recruiter_response_rate": 0.85,
            "interview_completion_rate": 0.90,
            "offer_acceptance_rate": 0.70,
            "github_activity_score": 0.78,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }

    mock_features_clean = {
        "yoe": 7.2,
        "total_career_months": 82,
        "grad_year": 2017,
        "earliest_job_year": 2017,
        "has_overlapping_dates": False,
        "started_before_graduating": False,
        "duration_mismatch_count": 0,
        "pure_research_no_production": False,
        "ai_only_langchain_under_12mo": False,
        "no_hands_on_code_18mo": False,
        "consulting_only_career": False,
        "cv_speech_robotics_no_nlp": False,
    }

    mock_features_disq = {
        **mock_features_clean,
        "no_hands_on_code_18mo": True,
    }

    mock_features_honeypot = {
        **mock_features_clean,
        "has_overlapping_dates": True,
        "started_before_graduating": True,
    }

    mock_text_blob = " ".join([
        mock_candidate["profile"]["headline"],
        mock_candidate["profile"]["summary"],
        *[r["description"] for r in mock_candidate["career_history"]],
    ])

    # ---- Function 1: semantic_score ----------------------------------------
    sys.stdout.write("Function 1: semantic_score\n")
    ss = semantic_score(mock_cand_emb, mock_jd_emb)
    _check("similar vectors -> score close to 1", ss > 0.8)
    _check("dissimilar vectors -> score close to 0", semantic_score(mock_dissimilar_emb, mock_jd_emb) < 0.1)
    _check("zero vector -> 0.0", semantic_score(np.zeros(DIM), mock_jd_emb) == 0.0)
    _check("returns float in [0, 1]", 0.0 <= ss <= 1.0)
    sys.stdout.write("\n")

    # ---- Function 2: is_disqualified ----------------------------------------
    sys.stdout.write("Function 2: is_disqualified\n")
    _check("clean candidate → False", not is_disqualified(mock_candidate, mock_features_clean))
    _check("disq flag set → True", is_disqualified(mock_candidate, mock_features_disq))
    sys.stdout.write("\n")

    # ---- Function 3: honeypot_score -----------------------------------------
    sys.stdout.write("Function 3: honeypot_score\n")
    hp_clean = honeypot_score(mock_candidate, mock_features_clean)
    hp_dirty = honeypot_score(mock_candidate, mock_features_honeypot)
    _check("clean candidate → 0 flags", hp_clean == 0)
    _check("honeypot candidate → >= 2 flags", hp_dirty >= 2)

    # Test expert with <= 2 months
    expert_2mo_candidate = {**mock_candidate, "skills": [
        {"name": "Quantum ML", "proficiency_level": "expert", "duration_months": 1, "endorsements": 0}
    ]}
    _check("expert skill <= 2 months → flag +1", honeypot_score(expert_2mo_candidate, mock_features_clean) >= 1)
    sys.stdout.write("\n")

    # ---- Function 4: jd_requirement_coverage --------------------------------
    sys.stdout.write("Function 4: jd_requirement_coverage\n")
    coverage_val = jd_requirement_coverage(mock_text_blob, mock_candidate["skills"])
    _check("non-zero coverage for relevant profile", coverage_val > 0.0)
    _check("coverage in [0, 1]", 0.0 <= coverage_val <= 1.0)
    _check("empty blob + no skills → 0.0", jd_requirement_coverage("", []) == 0.0)
    sys.stdout.write("\n")

    # ---- Function 5: skill_authenticity -------------------------------------
    sys.stdout.write("Function 5: skill_authenticity\n")
    auth = skill_authenticity(mock_candidate["skills"])
    _check("authentic skills → score in [0, 1]", 0.0 <= auth <= 1.0)
    _check("empty skills → 0.5", skill_authenticity([]) == 0.5)
    with_assessment = skill_authenticity(
        mock_candidate["skills"],
        assessment_scores={"python": 90.0, "faiss": 75.0}
    )
    _check("assessment scores boost authenticity", with_assessment >= auth or with_assessment > 0.5)
    sys.stdout.write("\n")

    # ---- Function 6: trajectory_score ---------------------------------------
    sys.stdout.write("Function 6: trajectory_score\n")
    traj = trajectory_score(mock_role_embs, mock_jd_emb)
    _check("trajectory score in [0, 1]", 0.0 <= traj <= 1.0)
    _check("non-zero for relevant roles", traj > 0.0)
    _check("empty role list → 0.0", trajectory_score([], mock_jd_emb) == 0.0)
    _check("zero JD embedding → 0.0", trajectory_score(mock_role_embs, np.zeros(DIM)) == 0.0)
    sys.stdout.write("\n")

    # ---- Function 7: behavioral_score ---------------------------------------
    sys.stdout.write("Function 7: behavioral_score\n")
    beh = behavioral_score(mock_candidate["redrob_signals"])
    _check("behavioral score in [0, 1]", 0.0 <= beh <= 1.0)
    _check("good signals → score > 0.5", beh > 0.5)
    _check("empty signals → 0.0", behavioral_score({}) == 0.0)

    # Inactivity penalty
    inactive_signals = {**mock_candidate["redrob_signals"], "last_active_date": "2020-01-01"}
    active_beh = behavioral_score(mock_candidate["redrob_signals"])
    inactive_beh = behavioral_score(inactive_signals)
    _check("inactive > 180 days → lower score", inactive_beh < active_beh)
    sys.stdout.write("\n")

    # ---- Function 8: title_shows_escalation ---------------------------------
    sys.stdout.write("Function 8: title_shows_escalation\n")
    _check("Junior → Senior → Senior ML → True", title_shows_escalation(mock_candidate["career_history"]))
    _check("single role → False", not title_shows_escalation([mock_candidate["career_history"][0]]))
    flat_history = [
        {"title": "Senior Engineer", "start_date": "2017-06"},
        {"title": "Senior Engineer", "start_date": "2020-06"},
    ]
    _check("flat titles → False", not title_shows_escalation(flat_history))
    sys.stdout.write("\n")

    # ---- Function 9: anti_pattern_penalty ------------------------------------
    sys.stdout.write("Function 9: anti_pattern_penalty\n")
    clean_penalty = anti_pattern_penalty(mock_candidate["career_history"], mock_text_blob)
    _check("clean candidate → 0.0 penalty", clean_penalty == 0.0)
    _check("penalty in [0, MAX]", 0.0 <= clean_penalty <= MAX_ANTI_PATTERN_PENALTY)

    # Title chaser scenario
    title_chaser_history = [
        {"title": "Junior Engineer", "company": "A", "start_date": "2020-01", "duration_months": 6},
        {"title": "Senior Engineer", "company": "B", "start_date": "2020-07", "duration_months": 8},
        {"title": "Staff Engineer", "company": "C", "start_date": "2021-03", "duration_months": 10},
    ]
    chaser_penalty = anti_pattern_penalty(title_chaser_history, "built some code")
    _check("title chaser → penalty >= 0.15", chaser_penalty >= 0.15)

    # Framework enthusiast scenario
    fw_blob = "built a chatbot tutorial poc toy project"
    fw_penalty = anti_pattern_penalty([], fw_blob)
    _check("framework enthusiast → penalty >= 0.10", fw_penalty >= 0.10)

    # Cap at MAX
    combined_blob = "built a chatbot tutorial poc toy project"
    capped = anti_pattern_penalty(title_chaser_history, combined_blob)
    _check("penalty capped at MAX_ANTI_PATTERN_PENALTY", capped <= MAX_ANTI_PATTERN_PENALTY)
    sys.stdout.write("\n")

    # ---- Function 10: flag_duplicates ----------------------------------------
    sys.stdout.write("Function 10: flag_duplicates\n")
    base_emb = RNG.random(DIM).astype(np.float32)
    near_dup_emb = base_emb + RNG.random(DIM).astype(np.float32) * 1e-5  # Very close
    different_emb = RNG.random(DIM).astype(np.float32)
    embs_matrix = np.stack([base_emb, near_dup_emb, different_emb])
    ids = ["A", "B", "C"]

    flagged = flag_duplicates(embs_matrix, ids)
    _check("near-duplicate pair detected", "A" in flagged or "B" in flagged)
    _check("distinct candidate not flagged", "C" not in flagged)

    # With behavioral scores: weaker one flagged
    flagged_beh = flag_duplicates(embs_matrix, ids, behavioral_scores=[0.9, 0.3, 0.8])
    _check("weaker behavioral score flagged (B)", "B" in flagged_beh)
    _check("stronger behavioral score kept (A not flagged)", "A" not in flagged_beh)

    _check("empty embeddings → empty set", flag_duplicates(np.empty((0, DIM)), []) == set())
    sys.stdout.write("\n")

    # ---- Function 11: final_score --------------------------------------------
    sys.stdout.write("Function 11: final_score\n")
    fs = final_score(
        semantic_fit=ss,
        coverage=coverage_val,
        trajectory=traj,
        authenticity=auth,
        behavioral=beh,
        anti_pattern_penalty_value=clean_penalty,
        is_disq=False,
        honeypot_flags=0,
    )
    _check("final score in [0, 1]", 0.0 <= fs <= 1.0)
    _check("disqualified → 0.0", final_score(0.9, 0.9, 0.9, 0.9, 0.9, 0.0, True, 0) == 0.0)
    _check("honeypot flags >= 2 → 0.0", final_score(0.9, 0.9, 0.9, 0.9, 0.9, 0.0, False, 3) == 0.0)
    _check("rounded to 4 dp", len(str(fs).split(".")[-1]) <= 4)
    _check("good candidate → positive score", fs > 0.0)
    sys.stdout.write("\n")

    # ---- Function 12: generate_reasoning -------------------------------------
    sys.stdout.write("Function 12: generate_reasoning\n")
    mock_scores = {
        "semantic_fit": ss,
        "coverage": coverage_val,
        "trajectory": traj,
        "authenticity": auth,
        "behavioral": beh,
        "final": fs,
    }
    reason = generate_reasoning(mock_candidate, mock_scores)
    _check("reasoning is non-empty string", isinstance(reason, str) and len(reason) > 0)
    _check("reasoning mentions candidate title", "Senior ML Engineer" in reason or "Engineer" in reason)
    _check("reasoning mentions years of experience", "7.2" in reason or "years" in reason)
    _check("no generic 'strong fit' template", "Candidate is a strong fit" not in reason)
    sys.stdout.write(f"\n  Reasoning: {reason}\n\n")

    # ---- Integration: prepare.py flattened schema ----------------------------
    sys.stdout.write("Integration: prepare.py flattened schema\n")

    flat_features = {
        "yoe": 7.2,
        "grad_year": 2017,
        "earliest_job_year": 2017,
        "total_career_months": 82,
        "has_overlapping_dates": False,
        "started_before_graduating": False,
        "duration_mismatch_count": 0,
        "skills": [
            {"name": "Python", "proficiency_level": "expert",
             "duration_months": 84, "endorsements": 30},
            {"name": "FAISS", "proficiency_level": "advanced",
             "duration_months": 36, "endorsements": 12},
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "last_active_date": "2026-06-01",
            "recruiter_response_rate": 0.85,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
        "current_title": "Senior ML Engineer",
        "current_company": "TechCorp AI",
    }

    _check("flat: is_disqualified False (missing flags)",
           not is_disqualified({}, flat_features))

    detail = disqualification_detail({}, flat_features)
    _check("flat: detail shows all flags as None",
           all(v is None for v in detail["flags_checked"].values()))

    hp_flat = honeypot_score(flat_features, flat_features)
    _check("flat: honeypot_score runs", isinstance(hp_flat, int))

    cov_no_blob = jd_requirement_coverage(None, flat_features.get("skills", []))
    _check("None text_blob: coverage from skills only",
           0.0 <= cov_no_blob <= 1.0)

    pen_none = anti_pattern_penalty(None, None)
    _check("None career + text_blob: penalty 0.0", pen_none == 0.0)

    beh_empty = behavioral_score({})
    _check("empty signals: score 0.0", beh_empty == 0.0)

    beh_nv = behavioral_score({"recruiter_response_rate": None})
    _check("None-valued signals: no crash", isinstance(beh_nv, float))

    flat_scores = {
        "semantic_fit": 0.82, "coverage": 0.75, "trajectory": 0.68,
        "authenticity": 0.91, "behavioral": 0.60, "final": 0.73,
    }
    reason_flat = generate_reasoning(flat_features, flat_scores)
    _check("flat schema: reasoning mentions title",
           "Senior ML Engineer" in reason_flat)
    _check("flat schema: reasoning mentions YOE", "7.2" in reason_flat)
    _check("flat schema: non-empty reasoning", len(reason_flat) > 20)

    reason_empty = generate_reasoning({}, {"final": 0.1})
    _check("empty candidate: reasoning no crash",
           isinstance(reason_empty, str))

    auth_nv = skill_authenticity([{"name": None, "proficiency_level": None}])
    _check("None-filled skill: no crash", isinstance(auth_nv, float))

    sem_fl = semantic_score(mock_cand_emb, mock_jd_emb)
    disq_fl = is_disqualified(flat_features, flat_features)
    hp_fl = honeypot_score(flat_features, flat_features)
    cov_fl = jd_requirement_coverage(None, flat_features.get("skills", []))
    auth_fl = skill_authenticity(flat_features.get("skills", []))
    traj_fl = trajectory_score(mock_role_embs, mock_jd_emb)
    beh_fl = behavioral_score(flat_features.get("redrob_signals", {}))
    pen_fl = anti_pattern_penalty(None, None)
    fs_fl = final_score(
        sem_fl, cov_fl, traj_fl, auth_fl, beh_fl, pen_fl, disq_fl, hp_fl,
    )
    _check("full pipeline on flat: score in [0,1]",
           0.0 <= fs_fl <= 1.0)
    _check("full pipeline on flat: non-zero", fs_fl > 0.0)
    sys.stdout.write("\n")

    # ---- Summary -------------------------------------------------------------
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    sys.stdout.write(f"Results: {passed}/{total} checks passed\n")
    if passed < total:
        sys.stdout.write("FAILED checks:\n")
        for name, ok in results:
            if not ok:
                sys.stdout.write(f"  - {name}\n")
        sys.exit(1)
    else:
        sys.stdout.write("All checks passed.\n")
