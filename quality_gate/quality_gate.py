"""
Quality Gate
HITL-Ops: Human-in-the-Loop ML Operations Platform

Role:
    Evaluates ML model metrics and requires explicit human approval
    before any model is deployed to production. If rejected, triggers
    automatic rollback to the last known-good model.

    This is the final checkpoint before a model goes live.

    Gate flow:
        1. Load model metrics (accuracy, F1, drift, etc.)
        2. Run automated checks against thresholds
        3. Generate evaluation report
        4. Present to human engineer via browser UI
        5. Human approves or rejects
        6. If approved  → model marked as deployment-ready
           If rejected  → rollback triggered automatically

Components:
    model_evaluator.py  → evaluates model metrics (this file)
    deployment_gate.py  → browser UI for human approval
    rollback.py         → handles rollback on rejection

Output:
    results/approved/   → approved model manifests
    results/rejected/   → rejected model manifests
    results/audit_trail.json → full decision log

Author: Prateek Gaur
Project: hitl-ops
"""

import json
import logging
import hashlib
import threading
import webbrowser
from pathlib import Path
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import parse_qs

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [QualityGate] %(levelname)s — %(message)s",
)
log = logging.getLogger("QualityGate")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parent.parent
RESULTS_DIR  = BASE_DIR / "results"
APPROVED_DIR = RESULTS_DIR / "approved"
REJECTED_DIR = RESULTS_DIR / "rejected"
AUDIT_FILE   = RESULTS_DIR / "audit_trail.json"
MODELS_DIR   = BASE_DIR / "data" / "sample_models"

PORT = 8766


# ── Metric thresholds ─────────────────────────────────────────────────────────

THRESHOLDS = {
    "accuracy":          {"min": 0.80, "warn": 0.85},
    "f1_score":          {"min": 0.75, "warn": 0.82},
    "precision":         {"min": 0.75, "warn": 0.82},
    "recall":            {"min": 0.75, "warn": 0.82},
    "drift_score":       {"max": 0.15, "warn": 0.10},   # lower is better
    "false_positive_rate": {"max": 0.20, "warn": 0.12},
    "latency_ms":        {"max": 500,  "warn": 300},
    "calibration_error": {"max": 0.10, "warn": 0.06},
}


# ── Model Evaluator ───────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluates a model's metrics against deployment thresholds.
    Generates a structured evaluation report.
    """

    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or THRESHOLDS

    def evaluate(self, metrics: Dict[str, float], model_name: str) -> Dict[str, Any]:
        """
        Evaluate model metrics against thresholds.
        Returns a full evaluation report.
        """
        checks      = []
        passed      = 0
        warned      = 0
        failed      = 0

        for metric, value in metrics.items():
            if metric not in self.thresholds:
                continue

            thresh  = self.thresholds[metric]
            check   = self._check_metric(metric, value, thresh)
            checks.append(check)

            if check["status"] == "pass":
                passed += 1
            elif check["status"] == "warn":
                warned += 1
            else:
                failed += 1

        # Overall gate decision
        if failed > 0:
            gate_result = "FAIL"
            gate_reason = f"{failed} metric(s) below minimum threshold"
        elif warned > 0:
            gate_result = "WARN"
            gate_reason = f"{warned} metric(s) in warning zone — human review recommended"
        else:
            gate_result = "PASS"
            gate_reason = "All metrics within acceptable range"

        # Overall score (0-100)
        total  = len(checks)
        score  = int(((passed + warned * 0.5) / total * 100)) if total else 0

        report = {
            "model_name":    model_name,
            "model_id":      self._model_id(model_name),
            "evaluated_at":  datetime.now().isoformat(),
            "metrics":       metrics,
            "checks":        checks,
            "passed":        passed,
            "warned":        warned,
            "failed":        failed,
            "total_checks":  total,
            "gate_result":   gate_result,
            "gate_reason":   gate_reason,
            "overall_score": score,
            "auto_approved": gate_result == "PASS" and failed == 0 and warned == 0,
        }

        log.info(
            f"Evaluation complete — model={model_name} | "
            f"result={gate_result} | score={score} | "
            f"passed={passed} warned={warned} failed={failed}"
        )
        return report

    def _check_metric(
        self, metric: str, value: float, thresh: Dict
    ) -> Dict[str, Any]:
        """Check a single metric against its threshold."""
        status  = "pass"
        message = ""

        if "min" in thresh:
            if value < thresh["min"]:
                status  = "fail"
                message = f"{value:.4f} below minimum {thresh['min']}"
            elif value < thresh.get("warn", thresh["min"]):
                status  = "warn"
                message = f"{value:.4f} in warning zone (warn={thresh.get('warn')})"
            else:
                message = f"{value:.4f} ✓"

        elif "max" in thresh:
            if value > thresh["max"]:
                status  = "fail"
                message = f"{value:.4f} above maximum {thresh['max']}"
            elif value > thresh.get("warn", thresh["max"]):
                status  = "warn"
                message = f"{value:.4f} in warning zone (warn={thresh.get('warn')})"
            else:
                message = f"{value:.4f} ✓"

        return {
            "metric":    metric,
            "value":     value,
            "threshold": thresh,
            "status":    status,
            "message":   message,
        }

    def _model_id(self, name: str) -> str:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        h   = hashlib.md5(name.encode()).hexdigest()[:6]
        return f"model_{ts}_{h}"


# ── Rollback handler ──────────────────────────────────────────────────────────

class RollbackHandler:
    """Handles model rollback on rejection."""

    def __init__(self):
        self.rollback_log: List[Dict] = []

    def trigger(self, model_id: str, reason: str, previous_model: str = "v_last_stable") -> Dict:
        """Trigger rollback to previous stable model."""
        event = {
            "event":          "rollback",
            "rejected_model": model_id,
            "rolled_back_to": previous_model,
            "reason":         reason,
            "triggered_at":   datetime.now().isoformat(),
        }
        self.rollback_log.append(event)
        log.warning(f"ROLLBACK triggered — {model_id} → {previous_model} | reason: {reason}")
        return event


# ── Audit logger ──────────────────────────────────────────────────────────────

def log_audit(event: Dict):
    """Append event to the audit trail."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trail = []
    if AUDIT_FILE.exists():
        with open(AUDIT_FILE) as f:
            try:
                trail = json.load(f)
            except Exception:
                trail = []
    trail.append(event)
    with open(AUDIT_FILE, "w") as f:
        json.dump(trail, f, indent=2)


def save_result(report: Dict, human_decision: Dict, output_dir: Path):
    """Save the full gate result (evaluation + human decision)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id  = report["model_id"]
    out_path  = output_dir / f"{model_id}.json"
    result    = {**report, "human_decision": human_decision}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Result saved → {out_path}")
    return str(out_path)


# ── HTML rendering ────────────────────────────────────────────────────────────

STATUS_COLOR = {"pass": "#16a34a", "warn": "#eab308", "fail": "#ef4444"}
STATUS_ICON  = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
GATE_COLOR   = {"PASS": "#16a34a", "WARN": "#eab308", "FAIL": "#ef4444"}


def render_gate_page(report: Dict, completed: Optional[Dict] = None) -> str:
    model_name   = report["model_name"]
    gate_result  = report["gate_result"]
    gate_reason  = report["gate_reason"]
    score        = report["overall_score"]
    checks       = report["checks"]
    metrics      = report["metrics"]
    evaluated_at = report["evaluated_at"]

    g_color = GATE_COLOR.get(gate_result, "#6b7280")

    # Metric rows
    metric_rows = ""
    for check in checks:
        s_color = STATUS_COLOR.get(check["status"], "#6b7280")
        s_icon  = STATUS_ICON.get(check["status"], "❓")
        metric_rows += f"""
        <tr style="border-bottom:1px solid #f3f4f6;">
            <td style="padding:10px 12px;font-weight:500;">{check['metric']}</td>
            <td style="padding:10px 12px;font-family:monospace;">
                {check['value']:.4f}</td>
            <td style="padding:10px 12px;font-size:12px;color:#6b7280;">
                {json.dumps(check['threshold'])}</td>
            <td style="padding:10px 12px;">
                <span style="color:{s_color};font-weight:600;">
                    {s_icon} {check['status'].upper()}</span></td>
            <td style="padding:10px 12px;font-size:13px;color:#374151;">
                {check['message']}</td>
        </tr>"""

    # Completed banner
    done_banner = ""
    if completed:
        action     = completed.get("action", "")
        bg         = "#16a34a" if action == "approve" else "#ef4444"
        done_banner = f"""
        <div style="background:{bg};color:#fff;padding:12px 20px;
                    border-radius:6px;margin-bottom:16px;font-size:14px;">
            {'✅ APPROVED' if action == 'approve' else '❌ REJECTED'} —
            by {completed.get('reviewer','human')} at {completed.get('decided_at','')}
            <br><em>{completed.get('comment','')}</em>
        </div>"""

    disabled = 'style="opacity:0.4;pointer-events:none;"' if completed else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HITL-Ops | Quality Gate</title>
    <style>
        * {{ box-sizing:border-box; margin:0; padding:0; }}
        body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                background:#f3f4f6; color:#111827; }}
        .topbar {{ background:#1e293b;color:#fff;padding:12px 24px;
                   display:flex;align-items:center;gap:16px; }}
        .topbar h1 {{ font-size:18px;font-weight:700; }}
        .sub {{ font-size:12px;color:#94a3b8; }}
        .container {{ max-width:960px;margin:24px auto;padding:0 16px; }}
        .card {{ background:#fff;border:1px solid #e5e7eb;border-radius:8px;
                 padding:20px;margin-bottom:16px; }}
        table {{ width:100%;border-collapse:collapse; }}
        th {{ text-align:left;padding:8px 12px;font-size:12px;
              color:#6b7280;border-bottom:2px solid #e5e7eb;
              text-transform:uppercase;letter-spacing:0.05em; }}
        button {{ cursor:pointer;border:none;border-radius:6px;
                  font-size:14px;font-weight:600;padding:10px 24px; }}
    </style>
</head>
<body>

<div class="topbar">
    <div>
        <h1>🔁 HITL-Ops · Quality Gate</h1>
        <div class="sub">Model Deployment Approval · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
    </div>
</div>

<div class="container">

    {done_banner}

    <!-- Model header -->
    <div class="card">
        <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
            <div>
                <div style="font-size:20px;font-weight:700;">{model_name}</div>
                <div style="font-size:13px;color:#6b7280;margin-top:2px;">
                    ID: {report['model_id']} · Evaluated: {evaluated_at}</div>
            </div>
            <div style="margin-left:auto;text-align:center;">
                <div style="font-size:36px;font-weight:800;color:{g_color};">
                    {score}</div>
                <div style="font-size:11px;color:#6b7280;">QUALITY SCORE</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:22px;font-weight:700;color:{g_color};
                            background:{g_color}22;padding:8px 20px;border-radius:6px;">
                    {gate_result}</div>
                <div style="font-size:11px;color:#6b7280;margin-top:4px;">
                    GATE RESULT</div>
            </div>
        </div>
        <div style="margin-top:12px;padding:10px 14px;background:#f9fafb;
                    border-radius:6px;font-size:13px;color:#374151;">
            <strong>Automated assessment:</strong> {gate_reason}
        </div>

        <!-- Pass/warn/fail summary -->
        <div style="display:flex;gap:12px;margin-top:12px;">
            <div style="flex:1;background:#dcfce7;border-radius:6px;padding:10px;
                        text-align:center;">
                <div style="font-size:24px;font-weight:700;color:#16a34a;">
                    {report['passed']}</div>
                <div style="font-size:11px;color:#166534;">PASSED</div>
            </div>
            <div style="flex:1;background:#fef9c3;border-radius:6px;padding:10px;
                        text-align:center;">
                <div style="font-size:24px;font-weight:700;color:#854d0e;">
                    {report['warned']}</div>
                <div style="font-size:11px;color:#854d0e;">WARNED</div>
            </div>
            <div style="flex:1;background:#fee2e2;border-radius:6px;padding:10px;
                        text-align:center;">
                <div style="font-size:24px;font-weight:700;color:#ef4444;">
                    {report['failed']}</div>
                <div style="font-size:11px;color:#b91c1c;">FAILED</div>
            </div>
        </div>
    </div>

    <!-- Metric checks -->
    <div class="card">
        <h2 style="font-size:15px;font-weight:600;margin-bottom:14px;">
            📊 Metric Evaluation</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th><th>Value</th><th>Threshold</th>
                    <th>Status</th><th>Detail</th>
                </tr>
            </thead>
            <tbody>{metric_rows}</tbody>
        </table>
    </div>

    <!-- Human decision form -->
    <div class="card" {disabled}>
        <h2 style="font-size:15px;font-weight:600;margin-bottom:14px;">
            👤 Human Deployment Decision</h2>
        <p style="font-size:13px;color:#374151;margin-bottom:16px;">
            Review the metrics above. Your decision determines whether this model
            is deployed to production or rolled back.
        </p>

        <form method="POST" action="/decide">
            <input type="hidden" name="model_id" value="{report['model_id']}">
            <input type="hidden" name="model_name" value="{model_name}">
            <input type="hidden" name="gate_result" value="{gate_result}">
            <input type="hidden" name="score" value="{score}">

            <!-- Comment -->
            <div style="margin-bottom:14px;">
                <label style="font-size:12px;font-weight:600;color:#374151;
                              display:block;margin-bottom:4px;">
                    DECISION NOTES (optional)</label>
                <textarea name="comment" rows="2"
                    placeholder="Reason for approval or rejection..."
                    style="width:100%;padding:8px 10px;border:1px solid #d1d5db;
                           border-radius:4px;font-size:13px;resize:vertical;">
                </textarea>
            </div>

            <!-- Buttons -->
            <div style="display:flex;gap:12px;">
                <button type="submit" name="action" value="approve"
                    style="background:#16a34a;color:#fff;flex:1;">
                    ✅ Approve — Deploy to Production
                </button>
                <button type="submit" name="action" value="reject"
                    style="background:#ef4444;color:#fff;flex:1;">
                    ❌ Reject — Trigger Rollback
                </button>
            </div>
        </form>
    </div>

</div>
</body>
</html>"""


def render_result_page(action: str, model_name: str) -> str:
    is_approved = action == "approve"
    color  = "#16a34a" if is_approved else "#ef4444"
    icon   = "✅" if is_approved else "❌"
    title  = "Model Approved — Deploying!" if is_approved else "Model Rejected — Rolling Back!"
    sub    = "The model has been marked as deployment-ready." if is_approved \
             else "Rollback to the last stable model has been triggered."
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="2;url=/">
    <style>
        body {{ font-family:sans-serif;display:flex;align-items:center;
                justify-content:center;height:100vh;background:#f3f4f6; }}
        .box {{ background:#fff;padding:40px 56px;border-radius:12px;
                border:2px solid {color};text-align:center; }}
    </style>
</head>
<body>
    <div class="box">
        <div style="font-size:48px;">{icon}</div>
        <div style="font-size:20px;font-weight:700;color:{color};margin-top:12px;">
            {title}</div>
        <div style="font-size:13px;color:#6b7280;margin-top:8px;">{sub}</div>
        <div style="font-size:13px;color:#6b7280;margin-top:4px;">Model: {model_name}</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:12px;">Redirecting...</div>
    </div>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class GateHandler(BaseHTTPRequestHandler):

    report:    Dict = {}
    completed: Optional[Dict] = None
    rollback:  RollbackHandler = RollbackHandler()

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        html = render_gate_page(self.report, self.completed)
        self._send(html)

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length).decode("utf-8")
        params  = parse_qs(body)

        def get(k, d=""):
            return params.get(k, [d])[0]

        action     = get("action", "reject")
        model_id   = get("model_id")
        model_name = get("model_name")
        gate_result= get("gate_result")
        score      = int(get("score", "0"))
        comment    = get("comment", "")

        decision = {
            "action":      action,
            "model_id":    model_id,
            "model_name":  model_name,
            "gate_result": gate_result,
            "score":       score,
            "comment":     comment,
            "reviewer":    "human_engineer",
            "decided_at":  datetime.now().isoformat(),
        }

        GateHandler.completed = decision

        if action == "approve":
            out_path = save_result(self.report, decision, APPROVED_DIR)
            audit_event = {
                "event": "model_approved", **decision, "saved_to": out_path
            }
            log.info(f"✅ Model APPROVED — {model_name} | score={score}")
        else:
            out_path    = save_result(self.report, decision, REJECTED_DIR)
            rb_event    = self.rollback.trigger(model_id, comment or gate_result)
            audit_event = {
                "event": "model_rejected", **decision,
                "saved_to": out_path, "rollback": rb_event
            }
            log.warning(f"❌ Model REJECTED — {model_name} | rollback triggered")

        log_audit(audit_event)

        html = render_result_page(action, model_name)
        self._send(html)

    def _send(self, html: str):
        enc = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(enc)))
        self.end_headers()
        self.wfile.write(enc)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_gate(metrics: Dict[str, float], model_name: str, port: int = PORT):
    """
    Run the quality gate for a model.
    Evaluates metrics, opens browser UI, waits for human decision.
    """
    evaluator = ModelEvaluator()
    report    = evaluator.evaluate(metrics, model_name)

    log.info(f"Quality gate opening for: {model_name}")
    log.info(f"Gate result: {report['gate_result']} | Score: {report['overall_score']}")

    GateHandler.report    = report
    GateHandler.completed = None

    server = HTTPServer(("localhost", port), GateHandler)
    url    = f"http://localhost:{port}"

    log.info(f"Quality gate UI → {url}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Quality gate closed")
        server.shutdown()

    return GateHandler.completed


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("HITL-Ops | Quality Gate")
    print("=" * 60)

    # Simulate 3 model scenarios

    # Scenario 1: Good model — should PASS
    print("\n[Scenario 1] Well-performing model")
    metrics_good = {
        "accuracy":          0.923,
        "f1_score":          0.901,
        "precision":         0.889,
        "recall":            0.914,
        "drift_score":       0.042,
        "false_positive_rate": 0.087,
        "latency_ms":        145.0,
        "calibration_error": 0.031,
    }

    # Scenario 2: Borderline model — should WARN
    print("[Scenario 2] Borderline model (some warnings)")
    metrics_warn = {
        "accuracy":          0.832,
        "f1_score":          0.798,
        "precision":         0.811,
        "recall":            0.786,
        "drift_score":       0.112,
        "false_positive_rate": 0.141,
        "latency_ms":        320.0,
        "calibration_error": 0.071,
    }

    # Scenario 3: Bad model — should FAIL
    print("[Scenario 3] Poor model (failures)")
    metrics_fail = {
        "accuracy":          0.731,
        "f1_score":          0.698,
        "precision":         0.712,
        "recall":            0.685,
        "drift_score":       0.221,
        "false_positive_rate": 0.287,
        "latency_ms":        612.0,
        "calibration_error": 0.143,
    }

    evaluator = ModelEvaluator()

    for name, metrics in [
        ("battery_anomaly_detector_v2.1", metrics_good),
        ("battery_anomaly_detector_v1.9", metrics_warn),
        ("battery_anomaly_detector_v1.5", metrics_fail),
    ]:
        report = evaluator.evaluate(metrics, name)
        print(f"\n  {name}")
        print(f"    Gate: {report['gate_result']:4s} | Score: {report['overall_score']:3d} | "
              f"Pass={report['passed']} Warn={report['warned']} Fail={report['failed']}")
        print(f"    Reason: {report['gate_reason']}")

    # Launch gate UI with the borderline model (most interesting for demo)
    print("\n" + "=" * 60)
    print("Launching Quality Gate UI for borderline model...")
    print("Opening http://localhost:8766")
    print("=" * 60)

    decision = run_gate(
        metrics    = metrics_warn,
        model_name = "battery_anomaly_detector_v1.9",
    )

    if decision:
        print(f"\nHuman decision: {decision['action'].upper()}")
        print(f"Comment: {decision.get('comment', '—')}")
        print(f"Audit trail updated: {AUDIT_FILE}")
