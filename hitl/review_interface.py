"""
Human Review Interface
HITL-Ops: Human-in-the-Loop ML Operations Platform

Role:
    Browser-based UI for human engineers to review flagged records
    from the confidence router. Humans can approve, correct, and rate
    each record. All decisions are logged for the RLHF feedback loop.

    Opens automatically in the default browser.
    Serves a local HTTP server — no external dependencies needed.

Features:
    - Live queue from router output (escalation + human_review)
    - Per-record: approve / correct label / rate quality (1-5) / comment
    - Priority-sorted (P0 first)
    - Severity badges, anomaly details, recommended actions
    - All human decisions saved to results/human_reviews.jsonl
    - Summary stats shown in real time

Usage:
    python hitl/review_interface.py
    # Opens http://localhost:8765 in browser

Author: Prateek Gaur
Project: hitl-ops
"""

import json
import logging
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ReviewInterface] %(levelname)s — %(message)s",
)
log = logging.getLogger("ReviewInterface")

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ESCALATION_F  = RESULTS_DIR / "router_escalation_queue.jsonl"
REVIEW_F      = RESULTS_DIR / "router_human_review.jsonl"
HUMAN_OUT     = RESULTS_DIR / "human_reviews.jsonl"

PORT = 8765


# ── Data layer ────────────────────────────────────────────────────────────────

def load_queue() -> List[Dict]:
    """Load escalation + human_review records, sorted by priority."""
    records = []
    for fpath in [ESCALATION_F, REVIEW_F]:
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    records.sort(key=lambda r: r.get("priority_int", 9))
    return records


def load_completed() -> Dict[str, Dict]:
    """Load already-reviewed records keyed by record_id."""
    completed = {}
    if HUMAN_OUT.exists():
        with open(HUMAN_OUT) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    completed[rec["record_id"]] = rec
    return completed


def save_review(review: Dict):
    """Append a human review to the output file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HUMAN_OUT, "a") as f:
        f.write(json.dumps(review) + "\n")
    log.info(f"Review saved — record={review['record_id']} action={review['human_action']}")


# ── HTML helpers ──────────────────────────────────────────────────────────────

SEVERITY_COLOR = {
    "critical": "#ef4444",
    "high":     "#f97316",
    "medium":   "#eab308",
    "low":      "#22c55e",
    "none":     "#6b7280",
}

PRIORITY_COLOR = {
    "P0": "#ef4444",
    "P1": "#f97316",
    "P2": "#eab308",
    "P3": "#3b82f6",
    "P4": "#22c55e",
}

DECISION_ICON = {
    "shutdown": "🔴",
    "escalate": "🟠",
    "review":   "🟡",
    "monitor":  "🔵",
    "ignore":   "⬜",
}


def badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:12px;font-weight:600;">{text}</span>'
    )


def render_record_card(rec: Dict, completed: Dict, idx: int) -> str:
    rid        = rec.get("record_id", "unknown")
    is_done    = rid in completed
    priority   = rec.get("priority", "P2")
    decision   = rec.get("decision", "review")
    label      = rec.get("label", "normal")
    confidence = rec.get("confidence", 0.0)
    severity   = rec.get("max_severity", "none")
    timestamp  = rec.get("timestamp", 0.0)
    summary    = rec.get("anomaly_summary", "")
    anomalies  = rec.get("anomalies", [])
    actions    = rec.get("recommended_actions", [])
    alert_msg  = rec.get("alert_message", "")
    conflict   = rec.get("conflict", False)

    p_color  = PRIORITY_COLOR.get(priority, "#6b7280")
    s_color  = SEVERITY_COLOR.get(severity, "#6b7280")
    d_icon   = DECISION_ICON.get(decision, "❓")
    done_cls = "opacity:0.5;pointer-events:none;" if is_done else ""

    # Completed badge
    done_banner = ""
    if is_done:
        rev = completed[rid]
        done_banner = f"""
        <div style="background:#16a34a;color:#fff;padding:6px 12px;
                    border-radius:4px;margin-bottom:8px;font-size:13px;">
            ✅ Reviewed — action: <strong>{rev['human_action']}</strong> |
            label: <strong>{rev['corrected_label']}</strong> |
            rating: <strong>{"⭐" * rev['quality_rating']}</strong>
        </div>"""

    # Alert banner
    alert_banner = ""
    if alert_msg:
        alert_banner = f"""
        <div style="background:#ef4444;color:#fff;padding:8px 12px;
                    border-radius:4px;margin-bottom:8px;font-size:13px;">
            🔔 {alert_msg}
        </div>"""

    # Conflict banner
    conflict_banner = ""
    if conflict:
        conflict_banner = f"""
        <div style="background:#7c3aed;color:#fff;padding:6px 12px;
                    border-radius:4px;margin-bottom:8px;font-size:13px;">
            ⚡ Classification conflict detected — review carefully
        </div>"""

    # Anomaly list
    anomaly_rows = ""
    for a in anomalies:
        a_color = SEVERITY_COLOR.get(a.get("severity", "none"), "#6b7280")
        anomaly_rows += f"""
        <tr>
            <td style="padding:4px 8px;">{badge(a.get('severity','?').upper(), a_color)}</td>
            <td style="padding:4px 8px;font-family:monospace;font-size:12px;">
                {a.get('rule','?')}</td>
            <td style="padding:4px 8px;font-size:13px;">{a.get('message','')}</td>
        </tr>"""

    # Actions list
    action_items = "".join(
        f'<li style="margin:3px 0;font-size:13px;">{a}</li>' for a in actions
    )

    # Sensor values
    soc   = rec.get("soc",        rec.get("pack_soc_mean", "—"))
    temp  = rec.get("temp_c",     rec.get("temp_mean_c",   "—"))
    volt  = rec.get("v_terminal", "—")
    r0    = rec.get("r0_eff",     "—")
    imbal = rec.get("imbalance",  "—")

    def fmt(v, unit="", decimals=3):
        if v == "—" or v is None:
            return "—"
        try:
            return f"{float(v):.{decimals}f}{unit}"
        except Exception:
            return str(v)

    sensor_row = f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin:8px 0;font-size:13px;">
        <span>⚡ <strong>V</strong>: {fmt(volt, 'V')}</span>
        <span>🔋 <strong>SOC</strong>: {fmt(soc, '', 3)}</span>
        <span>🌡 <strong>Temp</strong>: {fmt(temp, '°C', 1)}</span>
        <span>⚙️ <strong>R₀</strong>: {fmt(r0, 'Ω', 5)}</span>
        <span>⚖️ <strong>Imbalance</strong>: {fmt(imbal, '', 4)}</span>
        <span>🕐 <strong>t</strong>: {timestamp:.0f}s</span>
    </div>"""

    return f"""
    <div id="card-{rid}" style="border:1px solid #e5e7eb;border-radius:8px;
         padding:16px;margin-bottom:16px;background:#fff;{done_cls}">

        <!-- Header -->
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            {badge(priority, p_color)}
            {badge(severity.upper(), s_color)}
            {badge(decision.upper(), p_color)}
            <span style="font-family:monospace;font-size:12px;color:#6b7280;">
                #{idx+1} · {rid}</span>
            <span style="margin-left:auto;font-size:20px;">{d_icon}</span>
        </div>

        {done_banner}
        {alert_banner}
        {conflict_banner}

        <!-- Summary -->
        <div style="font-size:14px;color:#374151;margin-bottom:8px;">
            <strong>Summary:</strong> {summary}
        </div>

        <!-- Sensor values -->
        {sensor_row}

        <!-- Anomalies -->
        {"" if not anomalies else f'''
        <details style="margin:8px 0;">
            <summary style="cursor:pointer;font-size:13px;color:#374151;">
                🔍 Anomalies ({len(anomalies)})</summary>
            <table style="width:100%;margin-top:8px;border-collapse:collapse;">
                {anomaly_rows}
            </table>
        </details>'''}

        <!-- Recommended actions -->
        {"" if not actions else f'''
        <details style="margin:8px 0;">
            <summary style="cursor:pointer;font-size:13px;color:#374151;">
                📋 Recommended Actions</summary>
            <ul style="margin:8px 0 0 16px;">{action_items}</ul>
        </details>'''}

        <!-- Review form -->
        <div style="margin-top:12px;padding:12px;background:#f9fafb;
                    border-radius:6px;border:1px solid #e5e7eb;">
            <form method="POST" action="/review">
                <input type="hidden" name="record_id" value="{rid}">

                <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;">

                    <!-- Action -->
                    <div>
                        <label style="font-size:12px;font-weight:600;
                                      color:#374151;display:block;margin-bottom:4px;">
                            ACTION</label>
                        <select name="human_action"
                            style="padding:6px 10px;border:1px solid #d1d5db;
                                   border-radius:4px;font-size:13px;">
                            <option value="approve">✅ Approve</option>
                            <option value="correct">✏️ Correct</option>
                            <option value="escalate">🚨 Escalate</option>
                            <option value="dismiss">❌ Dismiss</option>
                        </select>
                    </div>

                    <!-- Corrected label -->
                    <div>
                        <label style="font-size:12px;font-weight:600;
                                      color:#374151;display:block;margin-bottom:4px;">
                            LABEL</label>
                        <select name="corrected_label"
                            style="padding:6px 10px;border:1px solid #d1d5db;
                                   border-radius:4px;font-size:13px;">
                            <option value="{label}" selected>{label}</option>
                            <option value="normal">normal</option>
                            <option value="warning">warning</option>
                            <option value="critical">critical</option>
                        </select>
                    </div>

                    <!-- Quality rating -->
                    <div>
                        <label style="font-size:12px;font-weight:600;
                                      color:#374151;display:block;margin-bottom:4px;">
                            QUALITY (1–5)</label>
                        <select name="quality_rating"
                            style="padding:6px 10px;border:1px solid #d1d5db;
                                   border-radius:4px;font-size:13px;">
                            <option value="5">⭐⭐⭐⭐⭐ Perfect</option>
                            <option value="4">⭐⭐⭐⭐ Good</option>
                            <option value="3" selected>⭐⭐⭐ OK</option>
                            <option value="2">⭐⭐ Poor</option>
                            <option value="1">⭐ Wrong</option>
                        </select>
                    </div>

                    <!-- Submit -->
                    <div>
                        <button type="submit"
                            style="padding:7px 18px;background:#2563eb;color:#fff;
                                   border:none;border-radius:4px;font-size:13px;
                                   font-weight:600;cursor:pointer;">
                            Submit Review
                        </button>
                    </div>
                </div>

                <!-- Comment -->
                <div style="margin-top:8px;">
                    <input type="text" name="comment" placeholder="Optional comment..."
                        style="width:100%;padding:6px 10px;border:1px solid #d1d5db;
                               border-radius:4px;font-size:13px;box-sizing:border-box;">
                </div>

            </form>
        </div>
    </div>"""


def render_page(queue: List[Dict], completed: Dict) -> str:
    n_total    = len(queue)
    n_done     = len(completed)
    n_pending  = n_total - n_done
    n_esc      = sum(1 for r in queue if r.get("priority") in ("P0","P1"))
    n_review   = sum(1 for r in queue if r.get("priority") == "P2")
    progress   = int((n_done / n_total * 100) if n_total else 0)

    cards = "".join(
        render_record_card(rec, completed, i)
        for i, rec in enumerate(queue)
    )

    if not queue:
        cards = """
        <div style="text-align:center;padding:60px;color:#6b7280;">
            <div style="font-size:48px;">✅</div>
            <div style="font-size:18px;margin-top:12px;">No records to review</div>
            <div style="font-size:13px;margin-top:4px;">
                Run the pipeline first to generate review records.</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HITL-Ops | Review Interface</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #f3f4f6; color: #111827; }}
        .topbar {{ background: #1e293b; color: #fff; padding: 12px 24px;
                   display: flex; align-items: center; gap: 16px; }}
        .topbar h1 {{ font-size: 18px; font-weight: 700; }}
        .topbar .sub {{ font-size: 12px; color: #94a3b8; }}
        .container {{ max-width: 900px; margin: 24px auto; padding: 0 16px; }}
        .stat-row {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
        .stat {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
                 padding: 12px 20px; flex: 1; min-width: 120px; text-align: center; }}
        .stat .num {{ font-size: 28px; font-weight: 700; }}
        .stat .lbl {{ font-size: 12px; color: #6b7280; margin-top: 2px; }}
        .progress-bar {{ background: #e5e7eb; border-radius: 4px;
                         height: 8px; margin-bottom: 20px; }}
        .progress-fill {{ background: #2563eb; border-radius: 4px;
                          height: 8px; width: {progress}%; transition: width 0.3s; }}
        details summary::-webkit-details-marker {{ display: none; }}
        details summary::before {{ content: "▶ "; font-size: 10px; }}
        details[open] summary::before {{ content: "▼ "; font-size: 10px; }}
    </style>
</head>
<body>

<div class="topbar">
    <div>
        <h1>🔁 HITL-Ops · Human Review Interface</h1>
        <div class="sub">Battery ECM Anomaly Review · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
    </div>
    <div style="margin-left:auto;">
        <a href="/" style="color:#94a3b8;font-size:13px;text-decoration:none;">↻ Refresh</a>
    </div>
</div>

<div class="container">

    <!-- Stats -->
    <div class="stat-row">
        <div class="stat">
            <div class="num" style="color:#ef4444;">{n_esc}</div>
            <div class="lbl">Escalations</div>
        </div>
        <div class="stat">
            <div class="num" style="color:#eab308;">{n_review}</div>
            <div class="lbl">For Review</div>
        </div>
        <div class="stat">
            <div class="num" style="color:#2563eb;">{n_pending}</div>
            <div class="lbl">Pending</div>
        </div>
        <div class="stat">
            <div class="num" style="color:#16a34a;">{n_done}</div>
            <div class="lbl">Completed</div>
        </div>
        <div class="stat">
            <div class="num">{progress}%</div>
            <div class="lbl">Progress</div>
        </div>
    </div>

    <!-- Progress bar -->
    <div class="progress-bar"><div class="progress-fill"></div></div>

    <!-- Cards -->
    {cards}

</div>

<script>
    // Auto-refresh every 30 seconds
    setTimeout(() => location.reload(), 30000);
</script>
</body>
</html>"""


def render_success(record_id: str, action: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="1;url=/">
    <title>Saved</title>
    <style>
        body {{ font-family: sans-serif; display: flex; align-items: center;
                justify-content: center; height: 100vh; background: #f3f4f6; }}
        .box {{ background: #fff; padding: 32px 48px; border-radius: 12px;
                border: 1px solid #e5e7eb; text-align: center; }}
    </style>
</head>
<body>
    <div class="box">
        <div style="font-size: 40px;">✅</div>
        <div style="font-size: 18px; font-weight: 600; margin-top: 12px;">
            Review saved!</div>
        <div style="font-size: 13px; color: #6b7280; margin-top: 6px;">
            Record {record_id} · Action: {action}</div>
        <div style="font-size: 12px; color: #9ca3af; margin-top: 8px;">
            Redirecting back...</div>
    </div>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class ReviewHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress default server logs

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            queue     = load_queue()
            completed = load_completed()
            html      = render_page(queue, completed)
            self._send_html(html)
        else:
            self._send_404()

    def do_POST(self):
        if self.path == "/review":
            length  = int(self.headers.get("Content-Length", 0))
            body    = self.rfile.read(length).decode("utf-8")
            params  = parse_qs(body)

            def get(key, default=""):
                vals = params.get(key, [default])
                return vals[0] if vals else default

            record_id       = get("record_id")
            human_action    = get("human_action", "approve")
            corrected_label = get("corrected_label", "normal")
            quality_rating  = int(get("quality_rating", "3"))
            comment         = get("comment", "")

            # Load original record for RLHF context
            queue     = load_queue()
            original  = next((r for r in queue if r.get("record_id") == record_id), {})

            review = {
                "record_id":         record_id,
                "human_action":      human_action,
                "corrected_label":   corrected_label,
                "original_label":    original.get("label", "unknown"),
                "quality_rating":    quality_rating,
                "comment":           comment,
                "was_corrected":     corrected_label != original.get("label", corrected_label),
                "priority":          original.get("priority", "unknown"),
                "decision":          original.get("decision", "unknown"),
                "anomaly_summary":   original.get("anomaly_summary", ""),
                "anomalies":         original.get("anomalies", []),
                "severity_score":    original.get("severity_score", 0.0),
                "confidence":        original.get("confidence", 0.0),
                "reviewed_at":       datetime.now().isoformat(),
                "reviewer":          "human_engineer",
            }

            save_review(review)

            html = render_success(record_id, human_action)
            self._send_html(html)
        else:
            self._send_404()

    def _send_html(self, html: str):
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_404(self):
        self.send_response(404)
        self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

def run(port: int = PORT, open_browser: bool = True):
    queue     = load_queue()
    completed = load_completed()

    log.info(f"Review queue loaded — {len(queue)} records | {len(completed)} already reviewed")

    if not queue:
        log.warning(
            "No records found. Run confidence_router.py first. "
            "Run the full pipeline first:\n"
            "  python hitl/confidence_router.py"
        )

    server = HTTPServer(("localhost", port), ReviewHandler)
    url    = f"http://localhost:{port}"

    log.info(f"Review interface running → {url}")
    log.info("Press Ctrl+C to stop")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down review interface")
        server.shutdown()


if __name__ == "__main__":
    run()
