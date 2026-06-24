"""
app.py — Redrob Candidate Intelligence Dashboard
=================================================
Streamlit recruiter dashboard for ranked_candidates.json output.

Usage:
    streamlit run app.py

Or point to a custom JSON:
    streamlit run app.py -- --ranked ./path/to/ranked_candidates.json

Requires:
    pip install streamlit pandas
"""

import argparse
import json
import os
import sys
import math

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Redrob · Candidate Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — dark terminal aesthetic ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    background-color: #0D1117;
    color: #C9D1D9;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] * { color: #C9D1D9 !important; }

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.kpi-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 18px 24px;
    flex: 1;
    min-width: 140px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #58A6FF;
    line-height: 1;
}
.kpi-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8B949E;
    margin-top: 6px;
}

/* ── Candidate row ── */
.cand-row {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    display: flex;
    align-items: center;
    gap: 16px;
}
.cand-row:hover { border-color: #58A6FF; background: #1C2230; }
.cand-row.selected { border-color: #58A6FF; background: #1C2230; }

.rank-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #8B949E;
    min-width: 40px;
    text-align: right;
}
.rank-badge.top10 { color: #58A6FF; font-weight: 600; }

.cand-name { font-weight: 500; font-size: 0.95rem; color: #E6EDF3; flex: 1; }
.cand-title { font-size: 0.8rem; color: #8B949E; flex: 1.5; }

.score-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
    min-width: 56px;
    text-align: center;
}
.score-high { background: #1A3A2A; color: #3FB950; border: 1px solid #2EA043; }
.score-mid  { background: #2D2A1A; color: #D29922; border: 1px solid #9E6A03; }
.score-low  { background: #2D1A1A; color: #F85149; border: 1px solid #DA3633; }

/* ── Evidence bar ── */
.evidence-bar-wrap { margin: 12px 0 4px; }
.evidence-bar {
    display: flex;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    background: #21262D;
}
.eb-seg { height: 100%; transition: width 0.3s; }
.eb-semantic  { background: #58A6FF; }
.eb-coverage  { background: #3FB950; }
.eb-trajectory{ background: #BC8CFF; }
.eb-auth      { background: #D29922; }
.eb-behavioral{ background: #39D5D5; }
.eb-legend {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 6px;
    font-size: 0.7rem;
    color: #8B949E;
}
.eb-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
}

/* ── Detail panel ── */
.detail-panel {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 24px;
}
.detail-header { font-size: 1.1rem; font-weight: 600; color: #E6EDF3; margin-bottom: 4px; }
.detail-sub { font-size: 0.82rem; color: #8B949E; margin-bottom: 20px; }

.reasoning-box {
    background: #0D1117;
    border-left: 3px solid #58A6FF;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
    color: #C9D1D9;
    line-height: 1.6;
    margin-bottom: 20px;
}

.check-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 0.82rem; }
.check-pass { color: #3FB950; }
.check-fail { color: #F85149; }
.check-warn { color: #D29922; }

.score-metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #21262D;
    font-size: 0.84rem;
}
.score-metric:last-child { border-bottom: none; }
.metric-label { color: #8B949E; }
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: #E6EDF3;
}

/* ── Section headers ── */
.section-eyebrow {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #58A6FF;
    margin-bottom: 10px;
    margin-top: 24px;
}

/* ── Tag pills ── */
.tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.tag {
    background: #21262D;
    border: 1px solid #30363D;
    color: #C9D1D9;
    font-size: 0.72rem;
    padding: 3px 9px;
    border-radius: 12px;
}
.tag.must { background: #1A3A2A; border-color: #2EA043; color: #3FB950; }
.tag.miss  { background: #2D1A1A; border-color: #DA3633; color: #F85149; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: #21262D;
    color: #C9D1D9;
    border: 1px solid #30363D;
    border-radius: 6px;
    font-size: 0.82rem;
}
.stButton > button:hover { border-color: #58A6FF; color: #58A6FF; }
div[data-testid="metric-container"] { background: #161B22; border: 1px solid #21262D; border-radius: 8px; padding: 12px; }
.stSlider > div > div { background: #21262D; }
.stSelectbox > div, .stMultiSelect > div { background: #161B22 !important; border: 1px solid #21262D !important; }
h1, h2, h3 { color: #E6EDF3 !important; }
hr { border-color: #21262D; }

/* Hide Streamlit default top padding */
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def find_ranked_file():
    """Look for ranked output in common locations."""
    candidates = [
        "ranked_candidates.json",
        "submission.csv",
        os.path.join("precomputed", "ranked_candidates.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load ranked_candidates.json into a DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for i, r in enumerate(raw):
        breakdown = r.get("_breakdown", r.get("breakdown", {}))
        profile = r.get("profile", {})

        rows.append({
            "rank":            i + 1,
            "candidate_id":    r.get("candidate_id", ""),
            "name":            r.get("name", r.get("candidate_id", "Unknown")),
            "title":           profile.get("current_title", r.get("title", "—")),
            "company":         profile.get("current_company", r.get("company", "—")),
            "yoe":             r.get("yoe", r.get("years_of_experience", None)),
            "final_score":     round(r.get("final_score", 0.0), 4),
            "semantic_fit":    round(breakdown.get("semantic_fit", 0.0), 4),
            "coverage":        round(breakdown.get("coverage", 0.0), 4),
            "trajectory":      round(breakdown.get("trajectory", 0.0), 4),
            "authenticity":    round(breakdown.get("authenticity", 0.0), 4),
            "behavioral":      round(breakdown.get("behavioral", 0.0), 4),
            "is_disqualified": r.get("is_disqualified", False),
            "honeypot_flags":  r.get("honeypot_flags", 0),
            "reasoning":       r.get("reasoning", ""),
        })

    df = pd.DataFrame(rows)
    return df


def score_class(score: float) -> str:
    if score >= 0.65:
        return "score-high"
    if score >= 0.40:
        return "score-mid"
    return "score-low"


def evidence_bar_html(row: dict) -> str:
    """Render a horizontal stacked evidence bar showing score breakdown."""
    weights = {
        "semantic_fit": 0.30,
        "coverage":     0.20,
        "trajectory":   0.15,
        "authenticity": 0.15,
        "behavioral":   0.20,
    }
    colors = {
        "semantic_fit": ("#58A6FF", "eb-semantic",   "Semantic Fit"),
        "coverage":     ("#3FB950", "eb-coverage",   "JD Coverage"),
        "trajectory":   ("#BC8CFF", "eb-trajectory", "Trajectory"),
        "authenticity": ("#D29922", "eb-auth",       "Authenticity"),
        "behavioral":   ("#39D5D5", "eb-behavioral", "Behavioral"),
    }
    total = sum(row.get(k, 0) * w for k, w in weights.items())
    total = max(total, 0.001)

    segs = ""
    for key, weight in weights.items():
        val = row.get(key, 0)
        pct = (val * weight / total) * 100
        color_hex, css_class, _ = colors[key]
        segs += f'<div class="eb-seg {css_class}" style="width:{pct:.1f}%"></div>'

    legend = ""
    for key, (color_hex, css_class, label) in colors.items():
        val = row.get(key, 0)
        legend += (f'<span><span class="eb-dot" style="background:{color_hex}"></span>'
                   f'{label}: <strong style="color:#C9D1D9">{val:.2f}</strong></span>')

    return f"""
    <div class="evidence-bar-wrap">
      <div class="evidence-bar">{segs}</div>
      <div class="eb-legend">{legend}</div>
    </div>
    """


# ── Sidebar filters ───────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("### 🔍 Redrob Ranker")
        st.markdown('<div style="font-size:0.75rem;color:#8B949E;margin-bottom:16px">Senior AI Engineer · Founding Team</div>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("**Filters**")

        show_disq = st.checkbox("Show disqualified profiles", value=False)
        show_honeypot = st.checkbox("Show honeypot-flagged profiles", value=False)

        min_score = st.slider("Minimum final score", 0.0, 1.0, 0.0, step=0.05)

        yoe_vals = df["yoe"].dropna()
        if len(yoe_vals) > 0:
            min_yoe, max_yoe = int(yoe_vals.min()), int(yoe_vals.max())
            if min_yoe < max_yoe:
                yoe_range = st.slider("Years of experience", min_yoe, max_yoe,
                                      (min_yoe, max_yoe))
            else:
                yoe_range = (min_yoe, max_yoe)
                st.info(f"YoE data: {min_yoe}y")
        else:
            yoe_range = (0, 99)

        top_n = st.selectbox("Show top N candidates", [10, 25, 50, 100], index=3)

        st.markdown("---")
        st.markdown("**Score weights (reference)**")
        weights_data = {
            "Semantic Fit":   "30%",
            "JD Coverage":    "20%",
            "Behavioral":     "20%",
            "Trajectory":     "15%",
            "Authenticity":   "15%",
        }
        for label, pct in weights_data.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.78rem;padding:3px 0;color:#8B949E">'
                f'<span>{label}</span><span style="color:#58A6FF;font-family:monospace">{pct}</span></div>',
                unsafe_allow_html=True,
            )

    return show_disq, show_honeypot, min_score, yoe_range, top_n


# ── KPI strip ─────────────────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame, filtered_df: pd.DataFrame):
    total = len(df)
    eligible = len(df[(~df["is_disqualified"]) & (df["honeypot_flags"] < 2)])
    disq_rate = len(df[df["is_disqualified"]]) / total * 100 if total else 0
    honeypot_rate = len(df[df["honeypot_flags"] >= 2]) / total * 100 if total else 0
    top10_avg = df.nsmallest(10, "rank")["final_score"].mean() if len(df) >= 10 else 0
    active_seekers = len(df[df["behavioral"] >= 0.5]) if "behavioral" in df.columns else 0

    cards = [
        (f"{total:,}",       "Total candidates"),
        (f"{eligible:,}",    "Eligible profiles"),
        (f"{disq_rate:.1f}%","Disqualification rate"),
        (f"{honeypot_rate:.1f}%", "Honeypot rate"),
        (f"{top10_avg:.3f}", "Avg score — Top 10"),
        (f"{active_seekers:,}", "Active seekers"),
    ]

    html = '<div class="kpi-row">'
    for val, label in cards:
        html += f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Candidate list ─────────────────────────────────────────────────────────────

def render_candidate_list(filtered_df: pd.DataFrame):
    st.markdown('<div class="section-eyebrow">Match Leaderboard</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No candidates match the current filters.")
        return None

    selected_idx = st.session_state.get("selected_idx", 0)

    for idx, row in filtered_df.iterrows():
        rank = row["rank"]
        rank_class = "top10" if rank <= 10 else ""
        sc = row["final_score"]
        sc_class = score_class(sc)
        name = row["name"] or row["candidate_id"]
        title = row["title"] or "—"
        is_selected = (idx == selected_idx)
        selected_class = "selected" if is_selected else ""

        col1, col2 = st.columns([8, 1])
        with col1:
            st.markdown(
                f'<div class="cand-row {selected_class}">'
                f'  <span class="rank-badge {rank_class}">#{rank}</span>'
                f'  <span class="cand-name">{name}</span>'
                f'  <span class="cand-title">{title}</span>'
                f'  <span class="score-pill {sc_class}">{sc:.3f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("View", key=f"view_{idx}"):
                st.session_state["selected_idx"] = idx
                st.rerun()

    return selected_idx


# ── Detail panel ──────────────────────────────────────────────────────────────

def render_detail_panel(df: pd.DataFrame, idx):
    if idx not in df.index:
        st.markdown(
            '<div class="detail-panel" style="text-align:center;color:#8B949E;padding:48px">'
            'Select a candidate to view their profile intelligence.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    row = df.loc[idx]
    sc = row["final_score"]
    sc_class = score_class(sc)

    st.markdown(
        f'<div class="detail-panel">'
        f'  <div class="detail-header">#{row["rank"]} — {row["name"] or row["candidate_id"]}</div>'
        f'  <div class="detail-sub">{row["title"]} · {row["company"]}'
        f'{"  ·  " + str(int(row["yoe"])) + " yrs exp" if pd.notna(row.get("yoe")) else ""}</div>',
        unsafe_allow_html=True,
    )

    # Score badge
    st.markdown(
        f'<span class="score-pill {sc_class}" style="font-size:1.1rem;padding:6px 16px">'
        f'Final Score: {sc:.4f}</span>',
        unsafe_allow_html=True,
    )

    # Evidence bar
    st.markdown(evidence_bar_html(row.to_dict()), unsafe_allow_html=True)

    # Reasoning
    st.markdown('<div class="section-eyebrow" style="margin-top:20px">Recruiter Reasoning</div>', unsafe_allow_html=True)
    reasoning = row["reasoning"] or "No reasoning generated."
    st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)

    # Score breakdown
    st.markdown('<div class="section-eyebrow">Score Breakdown</div>', unsafe_allow_html=True)
    metrics = [
        ("Semantic Fit",   row["semantic_fit"],  "30% weight · Cosine similarity vs JD embedding"),
        ("JD Coverage",    row["coverage"],      "20% weight · Must-have category coverage"),
        ("Behavioral",     row["behavioral"],    "20% weight · Availability & platform signals"),
        ("Trajectory",     row["trajectory"],    "15% weight · Career direction alignment"),
        ("Authenticity",   row["authenticity"],  "15% weight · Skill evidence cross-check"),
    ]
    st.markdown('<div style="background:#0D1117;border-radius:6px;padding:12px 16px">', unsafe_allow_html=True)
    for label, val, desc in metrics:
        bar_pct = int(val * 100)
        bar_color = "#58A6FF" if val >= 0.65 else "#D29922" if val >= 0.40 else "#F85149"
        st.markdown(
            f'<div class="score-metric">'
            f'  <div>'
            f'    <div class="metric-label">{label}</div>'
            f'    <div style="font-size:0.68rem;color:#484F58;margin-top:1px">{desc}</div>'
            f'  </div>'
            f'  <div style="display:flex;align-items:center;gap:10px">'
            f'    <div style="width:80px;height:4px;background:#21262D;border-radius:2px">'
            f'      <div style="width:{bar_pct}%;height:100%;background:{bar_color};border-radius:2px"></div>'
            f'    </div>'
            f'    <span class="metric-val">{val:.3f}</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Integrity audit
    st.markdown('<div class="section-eyebrow">Profile Integrity Audit</div>', unsafe_allow_html=True)
    checks = [
        ("✓", "check-pass", "Not disqualified")         if not row["is_disqualified"]  else ("✗", "check-fail", "Hard disqualification triggered"),
        ("✓", "check-pass", "Honeypot score clear")     if row["honeypot_flags"] == 0   else ("⚠", "check-warn", f"Honeypot flags: {int(row['honeypot_flags'])}"),
        ("✓", "check-pass", "Behavioral signals active") if row["behavioral"] >= 0.4    else ("⚠", "check-warn", "Low behavioral availability"),
        ("✓", "check-pass", "Skill authenticity solid") if row["authenticity"] >= 0.5  else ("⚠", "check-warn", "Skill authenticity concerns"),
        ("✓", "check-pass", "Trajectory aligned")       if row["trajectory"] >= 0.4    else ("⚠", "check-warn", "Career trajectory drift detected"),
    ]
    audit_html = ""
    for icon, css_class, text in checks:
        audit_html += f'<div class="check-row"><span class="{css_class}">{icon}</span><span style="font-size:0.82rem">{text}</span></div>'
    st.markdown(f'<div style="background:#0D1117;border-radius:6px;padding:12px 16px">{audit_html}</div>', unsafe_allow_html=True)

    # Candidate ID (for reference)
    st.markdown(
        f'<div style="margin-top:16px;font-size:0.72rem;color:#484F58;font-family:monospace">'
        f'ID: {row["candidate_id"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Find data file ─────────────────────────────────────────────────────────
    # Allow CLI override via sys.argv (streamlit run app.py -- --ranked path)
    ranked_path = None
    if "--ranked" in sys.argv:
        idx = sys.argv.index("--ranked")
        if idx + 1 < len(sys.argv):
            ranked_path = sys.argv[idx + 1]

    if ranked_path is None:
        ranked_path = find_ranked_file()

    if ranked_path is None or not os.path.exists(ranked_path):
        st.error(
            "**ranked_candidates.json not found.**\n\n"
            "Run `rank.py` first to generate it, then re-launch the dashboard.\n\n"
            "Expected location: `./ranked_candidates.json`"
        )
        st.stop()

    # ── Load ───────────────────────────────────────────────────────────────────
    with st.spinner("Loading candidate intelligence ..."):
        df = load_data(ranked_path)

    if df.empty:
        st.error("No candidates found in the ranked output file.")
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    show_disq, show_honeypot, min_score, yoe_range, top_n = render_sidebar(df)

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="font-size:1.4rem;font-weight:600;margin-bottom:2px">'
        '🔍 Candidate Intelligence · Senior AI Engineer</h1>'
        '<div style="font-size:0.78rem;color:#8B949E;margin-bottom:20px">'
        'Redrob AI · Founding Team Hire · Powered by DeepMatch</div>',
        unsafe_allow_html=True,
    )

    # ── KPI strip ──────────────────────────────────────────────────────────────
    render_kpis(df, df)

    # ── Apply filters ──────────────────────────────────────────────────────────
    filtered = df.copy()
    if not show_disq:
        filtered = filtered[~filtered["is_disqualified"]]
    if not show_honeypot:
        filtered = filtered[filtered["honeypot_flags"] < 2]
    filtered = filtered[filtered["final_score"] >= min_score]
    if "yoe" in filtered.columns:
        filtered = filtered[
            (filtered["yoe"].isna()) |
            ((filtered["yoe"] >= yoe_range[0]) & (filtered["yoe"] <= yoe_range[1]))
        ]
    filtered = filtered.nsmallest(top_n, "rank")

    # ── Two-column layout: list + detail ──────────────────────────────────────
    list_col, detail_col = st.columns([5, 4], gap="large")

    with list_col:
        selected_idx = render_candidate_list(filtered)

    with detail_col:
        st.markdown('<div class="section-eyebrow">Profile Intelligence</div>', unsafe_allow_html=True)
        current_idx = st.session_state.get("selected_idx", filtered.index[0] if not filtered.empty else None)
        render_detail_panel(df, current_idx)

    # ── Export strip ───────────────────────────────────────────────────────────
    st.markdown("---")
    export_col1, export_col2, export_col3 = st.columns([2, 2, 6])
    with export_col1:
        csv_data = filtered[["rank", "candidate_id", "name", "title", "final_score", "reasoning"]].to_csv(index=False)
        st.download_button(
            "⬇ Export filtered CSV",
            data=csv_data,
            file_name="redrob_filtered.csv",
            mime="text/csv",
        )
    with export_col2:
        json_data = filtered[["rank", "candidate_id", "name", "final_score", "reasoning"]].to_json(orient="records", indent=2)
        st.download_button(
            "⬇ Export JSON",
            data=json_data,
            file_name="redrob_filtered.json",
            mime="application/json",
        )
    with export_col3:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#8B949E;padding-top:8px">'
            f'Showing {len(filtered)} of {len(df)} total candidates · '
            f'Data: <code style="color:#58A6FF">{ranked_path}</code></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()