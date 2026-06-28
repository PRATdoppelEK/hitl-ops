"""
Streamlit Monitoring Dashboard
HITL-Ops: Human-in-the-Loop ML Operations Platform

Role:
    Live monitoring dashboard showing the full pipeline state:
    - Pipeline run stats (records processed, routed, reviewed)
    - Review queue progress (escalation / review / auto-approved)
    - Anomaly breakdown by type and severity
    - RLHF dataset growth and correction rate
    - Quality gate history (approved / rejected models)
    - Audit trail feed

Usage:
    streamlit run dashboard/app.py

Author: Prateek Gaur
Project: hitl-ops
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parent.parent
RESULTS_DIR  = BASE_DIR / "results"

ESCALATION_F = RESULTS_DIR / "router_escalation_queue.jsonl"
REVIEW_F     = RESULTS_DIR / "router_human_review.jsonl"
AUTO_F       = RESULTS_DIR / "router_auto_approve.jsonl"
MANIFEST_F   = RESULTS_DIR / "router_manifest.json"
REVIEWS_F    = RESULTS_DIR / "human_reviews.jsonl"
RLHF_F       = RESULTS_DIR / "rlhf_dataset.jsonl"
STATS_F      = RESULTS_DIR / "feedback_stats.json"
AUDIT_F      = RESULTS_DIR / "audit_trail.json"
APPROVED_DIR = RESULTS_DIR / "approved"
REJECTED_DIR = RESULTS_DIR / "rejected"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HITL-Ops Dashboard",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-num {
        font-size: 36px;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-lbl {
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .alert-box {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .section-header {
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records


@st.cache_data(ttl=10)
def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path) as f:
        try:
            return json.load(f)
        except Exception:
            return {}


@st.cache_data(ttl=10)
def load_model_results(directory: Path):
    if not directory.exists():
        return []
    results = []
    for f in directory.glob("*.json"):
        with open(f) as fp:
            try:
                results.append(json.load(fp))
            except Exception:
                continue
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔁 HITL-Ops")
    st.markdown("**Human-in-the-Loop ML Ops**")
    st.markdown("---")
    st.markdown(f"**Last refresh:** {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("- [Review Interface](http://localhost:8765)")
    st.markdown("- [Quality Gate](http://localhost:8766)")
    st.markdown("---")
    st.markdown("### 📁 Data Sources")
    for label, path in [
        ("Escalation queue", ESCALATION_F),
        ("Human review queue", REVIEW_F),
        ("Auto-approve queue", AUTO_F),
        ("Human reviews", REVIEWS_F),
        ("RLHF dataset", RLHF_F),
        ("Audit trail", AUDIT_F),
    ]:
        exists = "✅" if path.exists() else "❌"
        st.markdown(f"{exists} {label}")


# ── Load all data ─────────────────────────────────────────────────────────────

escalation  = load_jsonl(ESCALATION_F)
review_q    = load_jsonl(REVIEW_F)
auto_q      = load_jsonl(AUTO_F)
manifest    = load_json(MANIFEST_F)
reviews     = load_jsonl(REVIEWS_F)
rlhf        = load_jsonl(RLHF_F)
fb_stats    = load_json(STATS_F)
audit       = load_json(AUDIT_F)
approved    = load_model_results(APPROVED_DIR)
rejected    = load_model_results(REJECTED_DIR)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# 🔁 HITL-Ops Monitoring Dashboard")
st.markdown(
    f"Battery ECM Anomaly Pipeline · "
    f"Batch: `{manifest.get('batch_id', 'N/A')}` · "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
st.markdown("---")


# ── Section 1: Pipeline Overview ──────────────────────────────────────────────

st.markdown('<div class="section-header">📊 Pipeline Overview</div>',
            unsafe_allow_html=True)

total      = manifest.get("total_records", len(escalation) + len(review_q) + len(auto_q))
n_esc      = len(escalation)
n_rev      = len(review_q)
n_auto     = len(auto_q)
n_reviewed = len(reviews)
n_pending  = (n_esc + n_rev) - n_reviewed
progress   = int(n_reviewed / (n_esc + n_rev) * 100) if (n_esc + n_rev) > 0 else 0

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Records", total)
with col2:
    st.metric("🔴 Escalations", n_esc)
with col3:
    st.metric("🟡 For Review", n_rev)
with col4:
    st.metric("✅ Auto-Approved", n_auto)
with col5:
    st.metric("👤 Reviewed", n_reviewed)
with col6:
    st.metric("⏳ Pending", n_pending)

# Progress bar
st.markdown(f"**Human Review Progress: {progress}%**")
st.progress(progress / 100)

# Drift alerts
drift_alerts = manifest.get("drift_alerts", [])
if drift_alerts:
    st.markdown("**⚠️ Drift Alerts:**")
    for alert in drift_alerts:
        st.markdown(
            f'<div class="alert-box">⚠️ {alert}</div>',
            unsafe_allow_html=True
        )

st.markdown("---")


# ── Section 2: Routing Distribution ──────────────────────────────────────────

st.markdown('<div class="section-header">🔀 Routing Distribution</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Routing pie chart
    if total > 0:
        routing_data = pd.DataFrame({
            "Queue":  ["Auto-Approve", "Escalation", "Human Review"],
            "Count":  [n_auto, n_esc, n_rev],
            "Color":  ["#16a34a", "#ef4444", "#eab308"],
        })
        st.bar_chart(
            routing_data.set_index("Queue")["Count"],
            color=["#3b82f6"],
        )

with col2:
    # Rates
    auto_rate = manifest.get("auto_approve_rate", 0)
    esc_rate  = manifest.get("escalation_rate", 0)
    rev_rate  = manifest.get("review_rate", 0)

    st.markdown("**Routing Rates**")
    st.markdown(f"✅ Auto-approve: **{auto_rate:.1%}**")
    st.progress(auto_rate)
    st.markdown(f"🔴 Escalation: **{esc_rate:.1%}**")
    st.progress(esc_rate)
    st.markdown(f"🟡 Human review: **{rev_rate:.1%}**")
    st.progress(rev_rate)

st.markdown("---")


# ── Section 3: Anomaly Breakdown ─────────────────────────────────────────────

st.markdown('<div class="section-header">🔍 Anomaly Breakdown</div>',
            unsafe_allow_html=True)

all_queue = escalation + review_q
if all_queue:
    # Severity distribution
    severity_counts = {}
    rule_counts     = {}
    decision_counts = {}

    for rec in all_queue:
        sev = rec.get("max_severity", "none")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        dec = rec.get("decision", "unknown")
        decision_counts[dec] = decision_counts.get(dec, 0) + 1

        for a in rec.get("anomalies", []):
            rule = a.get("rule", "unknown")
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Severity Distribution**")
        sev_df = pd.DataFrame(
            list(severity_counts.items()),
            columns=["Severity", "Count"]
        ).sort_values("Count", ascending=False)
        st.dataframe(sev_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Decision Distribution**")
        dec_df = pd.DataFrame(
            list(decision_counts.items()),
            columns=["Decision", "Count"]
        ).sort_values("Count", ascending=False)
        st.dataframe(dec_df, hide_index=True, use_container_width=True)

    with col3:
        st.markdown("**Top Anomaly Rules**")
        rule_df = pd.DataFrame(
            list(rule_counts.items()),
            columns=["Rule", "Count"]
        ).sort_values("Count", ascending=False).head(8)
        st.dataframe(rule_df, hide_index=True, use_container_width=True)

    # Escalation queue detail
    st.markdown("**🔴 Escalation Queue**")
    if escalation:
        esc_rows = []
        for r in escalation:
            esc_rows.append({
                "Record ID":  r.get("record_id", "")[:14],
                "Priority":   r.get("priority", ""),
                "Decision":   r.get("decision", ""),
                "Severity":   r.get("max_severity", ""),
                "Confidence": f"{r.get('confidence', 0):.2f}",
                "Time (s)":   int(r.get("timestamp", 0)),
                "Summary":    r.get("anomaly_summary", "")[:60],
            })
        st.dataframe(pd.DataFrame(esc_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No escalation records.")

else:
    st.info("No pipeline data found. Run confidence_router.py first.")

st.markdown("---")


# ── Section 4: Human Review Stats ────────────────────────────────────────────

st.markdown('<div class="section-header">👤 Human Review Activity</div>',
            unsafe_allow_html=True)

if reviews:
    col1, col2, col3, col4 = st.columns(4)

    actions        = [r.get("human_action", "") for r in reviews]
    corrections    = [r for r in reviews if r.get("was_corrected")]
    ratings        = [r.get("quality_rating", 3) for r in reviews]
    avg_rating     = np.mean(ratings)

    with col1:
        st.metric("Total Reviews", len(reviews))
    with col2:
        st.metric("Corrections", len(corrections))
    with col3:
        st.metric("Correction Rate",
                  f"{len(corrections)/len(reviews):.1%}")
    with col4:
        st.metric("Avg Quality Rating", f"{avg_rating:.1f} ⭐")

    # Action breakdown
    col1, col2 = st.columns(2)
    with col1:
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        st.markdown("**Review Actions**")
        act_df = pd.DataFrame(
            list(action_counts.items()),
            columns=["Action", "Count"]
        )
        st.dataframe(act_df, hide_index=True, use_container_width=True)

    with col2:
        # Rating distribution
        st.markdown("**Quality Rating Distribution**")
        rating_counts = {i: ratings.count(i) for i in range(1, 6)}
        rating_df = pd.DataFrame(
            [(f"{'⭐'*k} ({k})", v) for k, v in rating_counts.items()],
            columns=["Rating", "Count"]
        )
        st.dataframe(rating_df, hide_index=True, use_container_width=True)

    # Recent reviews table
    st.markdown("**Recent Reviews**")
    review_rows = []
    for r in reviews[-10:][::-1]:
        review_rows.append({
            "Record":    r.get("record_id", "")[:14],
            "Action":    r.get("human_action", ""),
            "Original":  r.get("original_label", ""),
            "Corrected": r.get("corrected_label", ""),
            "Changed":   "✏️" if r.get("was_corrected") else "✅",
            "Rating":    "⭐" * r.get("quality_rating", 3),
            "Comment":   r.get("comment", "")[:40],
            "Time":      r.get("reviewed_at", "")[:19],
        })
    st.dataframe(pd.DataFrame(review_rows), hide_index=True, use_container_width=True)
else:
    st.info("No reviews yet. Open the review interface and submit some reviews.")

st.markdown("---")


# ── Section 5: RLHF Dataset ───────────────────────────────────────────────────

st.markdown('<div class="section-header">🧠 RLHF Feedback Dataset</div>',
            unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

n_rlhf         = len(rlhf)
corr_rate      = fb_stats.get("correction_rate", 0.0)
avg_reward     = fb_stats.get("reward_mean", 0.0)
avg_q_rating   = fb_stats.get("avg_quality_rating", 0.0)

with col1:
    st.metric("Training Records", n_rlhf)
with col2:
    st.metric("Correction Rate", f"{corr_rate:.1%}")
with col3:
    st.metric("Avg Reward", f"{avg_reward:.3f}")
with col4:
    st.metric("Avg Quality", f"{avg_q_rating:.1f} ⭐")

if rlhf:
    col1, col2 = st.columns(2)

    with col1:
        # Label correction map
        orig_dist = fb_stats.get("original_label_dist", {})
        corr_dist = fb_stats.get("corrected_label_dist", {})
        if orig_dist:
            st.markdown("**Original Label Distribution**")
            st.dataframe(
                pd.DataFrame(list(orig_dist.items()),
                             columns=["Label", "Count"]),
                hide_index=True, use_container_width=True
            )

    with col2:
        if corr_dist:
            st.markdown("**Corrected Label Distribution**")
            st.dataframe(
                pd.DataFrame(list(corr_dist.items()),
                             columns=["Label", "Count"]),
                hide_index=True, use_container_width=True
            )

    # Sample RLHF record
    with st.expander("📄 Sample RLHF Training Record"):
        sample = rlhf[0]
        st.markdown("**Prompt:**")
        st.code(sample.get("prompt", ""), language="text")
        st.markdown("**Chosen (correct):**")
        st.info(sample.get("chosen", ""))
        st.markdown("**Rejected (model error):**")
        st.warning(sample.get("rejected", ""))
        st.markdown(f"**Reward:** `{sample.get('reward', 0):.4f}`")

st.markdown("---")


# ── Section 6: Quality Gate History ──────────────────────────────────────────

st.markdown('<div class="section-header">🚦 Quality Gate History</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("✅ Approved Models", len(approved))
with col2:
    st.metric("❌ Rejected Models", len(rejected))
with col3:
    total_gate = len(approved) + len(rejected)
    approve_rate = len(approved) / total_gate if total_gate > 0 else 0
    st.metric("Approval Rate", f"{approve_rate:.1%}")

all_gate = approved + rejected
if all_gate:
    gate_rows = []
    for r in all_gate:
        hd = r.get("human_decision", {})
        gate_rows.append({
            "Model":      r.get("model_name", "")[:35],
            "Score":      r.get("overall_score", 0),
            "Gate":       r.get("gate_result", ""),
            "Decision":   hd.get("action", "").upper(),
            "Pass":       r.get("passed", 0),
            "Warn":       r.get("warned", 0),
            "Fail":       r.get("failed", 0),
            "Comment":    hd.get("comment", "")[:30],
            "Time":       hd.get("decided_at", "")[:19],
        })
    st.dataframe(
        pd.DataFrame(gate_rows).sort_values("Time", ascending=False),
        hide_index=True, use_container_width=True
    )
else:
    st.info("No quality gate decisions yet. Run quality_gate.py first.")

st.markdown("---")


# ── Section 7: Audit Trail ────────────────────────────────────────────────────

st.markdown('<div class="section-header">📋 Audit Trail</div>',
            unsafe_allow_html=True)

if audit and isinstance(audit, list):
    st.markdown(f"**{len(audit)} audit events recorded**")
    audit_rows = []
    for event in audit[-20:][::-1]:
        audit_rows.append({
            "Event":   event.get("event", ""),
            "Model":   event.get("model_name", event.get("model_id", ""))[:30],
            "Action":  event.get("action", ""),
            "Score":   event.get("score", ""),
            "Time":    event.get("decided_at", "")[:19],
        })
    st.dataframe(
        pd.DataFrame(audit_rows),
        hide_index=True, use_container_width=True
    )
else:
    st.info("No audit events yet. Approve or reject a model in the quality gate.")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:12px;'>"
    "HITL-Ops · Battery ECM Anomaly Pipeline · "
    f"Built by Prateek Gaur · {datetime.now().strftime('%Y-%m-%d')}"
    "</div>",
    unsafe_allow_html=True
)
