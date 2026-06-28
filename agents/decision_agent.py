"""
Agent 4 — Decision Agent
HITL-Ops: Human-in-the-Loop ML Operations Platform

Role:
    Final agent in the pipeline. Receives classified records from
    Agent 3 and produces a concrete, actionable decision for each.

    This is the last automated step before human review or auto-approval.

    Decision actions:
        "ignore"    → normal, high confidence — no action needed
        "monitor"   → normal/warning, watch for trends
        "review"    → send to human review interface
        "escalate"  → critical, alert + immediate human attention
        "shutdown"  → extreme critical — recommend immediate system shutdown

    Priority levels:
        P0 → immediate (shutdown / thermal runaway)
        P1 → urgent    (escalate / critical)
        P2 → standard  (review / warning)
        P3 → low       (monitor / normal)
        P4 → none      (ignore)

Output per record (adds to Agent 3 dict):
    {
        "decision":          str,    # "ignore" | "monitor" | "review" | "escalate" | "shutdown"
        "priority":          str,    # "P0" | "P1" | "P2" | "P3" | "P4"
        "priority_int":      int,    # 0–4 for sorting
        "action_reason":     str,    # why this decision was made
        "recommended_actions": list, # concrete steps the human should take
        "alert_triggered":   bool,   # True if P0 or P1
        "alert_message":     str,    # alert text (empty if no alert)
        "pipeline_complete": bool,   # True — marks end of agent pipeline
        "agent4_status":     str,    # "decided" | "error"
        "routing_target":    str,    # "auto_approve" | "human_review" | "escalation_queue"
    }

Author: Prateek Gaur
Project: hitl-ops
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DecisionAgent] %(levelname)s — %(message)s",
)
log = logging.getLogger("DecisionAgent")


# ── Decision rules ────────────────────────────────────────────────────────────

# Maps (label, confidence_band, conflict) → (decision, priority)
# Checked top-to-bottom, first match wins

DECISION_RULES = [

    # ── P0: Shutdown conditions ───────────────────────────────────────────────
    # Thermal runaway → immediate shutdown
    {
        "name":       "thermal_runaway_shutdown",
        "conditions": lambda r: (
            r.get("label") == "critical"
            and any(a["rule"] == "thermal_runaway_risk" for a in r.get("anomalies", []))
        ),
        "decision":   "shutdown",
        "priority":   "P0",
        "reason":     "Thermal runaway risk detected — immediate shutdown recommended",
        "actions": [
            "Immediately disconnect battery from load",
            "Activate cooling system if available",
            "Alert safety team",
            "Do NOT attempt to charge or discharge",
            "Inspect cell physically for swelling or damage",
        ],
    },

    # ── P1: Escalate conditions ───────────────────────────────────────────────
    # Critical + high confidence → escalate
    {
        "name":       "critical_high_conf_escalate",
        "conditions": lambda r: (
            r.get("label") == "critical"
            and r.get("confidence_band") == "high"
            and not any(a["rule"] == "thermal_runaway_risk" for a in r.get("anomalies", []))
        ),
        "decision":   "escalate",
        "priority":   "P1",
        "reason":     "Critical anomaly detected with high confidence — escalating for immediate review",
        "actions": [
            "Reduce load current immediately",
            "Check BMS fault codes",
            "Review last 10 minutes of sensor data",
            "Prepare for possible system shutdown",
        ],
    },

    # Critical + medium confidence → escalate with note
    {
        "name":       "critical_medium_conf_escalate",
        "conditions": lambda r: (
            r.get("label") == "critical"
            and r.get("confidence_band") == "medium"
        ),
        "decision":   "escalate",
        "priority":   "P1",
        "reason":     "Critical anomaly with medium confidence — escalating, human should verify",
        "actions": [
            "Verify sensor readings against redundant sensors",
            "Check for wiring or sensor faults",
            "If confirmed, reduce load immediately",
            "Log incident for investigation",
        ],
    },

    # Critical + low confidence → review first
    {
        "name":       "critical_low_conf_review",
        "conditions": lambda r: (
            r.get("label") == "critical"
            and r.get("confidence_band") == "low"
        ),
        "decision":   "review",
        "priority":   "P2",
        "reason":     "Critical label with low confidence — possible sensor fault, human review required",
        "actions": [
            "Check sensor calibration",
            "Compare with adjacent cell readings",
            "Determine if anomaly is real or measurement artifact",
        ],
    },

    # ── P2: Review conditions ─────────────────────────────────────────────────
    # Warning + conflict → review
    {
        "name":       "warning_conflict_review",
        "conditions": lambda r: (
            r.get("label") == "warning"
            and r.get("conflict") is True
        ),
        "decision":   "review",
        "priority":   "P2",
        "reason":     "Warning with classification conflict — human review needed to resolve ambiguity",
        "actions": [
            "Review the conflicting anomaly signals",
            "Check if trend is worsening over time",
            "Decide: continue monitoring or escalate",
        ],
    },

    # Warning + medium/low confidence → review
    {
        "name":       "warning_uncertain_review",
        "conditions": lambda r: (
            r.get("label") == "warning"
            and r.get("confidence_band") in ("medium", "low")
        ),
        "decision":   "review",
        "priority":   "P2",
        "reason":     "Warning with uncertain confidence — human review recommended",
        "actions": [
            "Verify anomaly is real and not transient",
            "Check system operating conditions",
            "Decide on monitoring interval",
        ],
    },

    # Warning + high confidence → monitor
    {
        "name":       "warning_high_conf_monitor",
        "conditions": lambda r: (
            r.get("label") == "warning"
            and r.get("confidence_band") == "high"
        ),
        "decision":   "monitor",
        "priority":   "P3",
        "reason":     "Warning detected with high confidence — monitoring recommended",
        "actions": [
            "Increase sampling frequency for this cell/pack",
            "Set alert if metric worsens by >10%",
            "Schedule inspection at next maintenance window",
        ],
    },

    # ── P3: Monitor conditions ────────────────────────────────────────────────
    # Normal + conflict → monitor (edge case)
    {
        "name":       "normal_conflict_monitor",
        "conditions": lambda r: (
            r.get("label") == "normal"
            and r.get("conflict") is True
        ),
        "decision":   "monitor",
        "priority":   "P3",
        "reason":     "Normal label but classification conflict detected — monitoring recommended",
        "actions": [
            "Watch for trend development over next few cycles",
            "No immediate action required",
        ],
    },

    # ── P4: Ignore conditions ─────────────────────────────────────────────────
    # Normal + high confidence + no conflict → ignore
    {
        "name":       "normal_clean_ignore",
        "conditions": lambda r: (
            r.get("label") == "normal"
            and r.get("confidence_band") == "high"
            and not r.get("conflict")
        ),
        "decision":   "ignore",
        "priority":   "P4",
        "reason":     "Normal classification with high confidence — no action required",
        "actions":    [],
    },

    # Normal + medium confidence → monitor
    {
        "name":       "normal_medium_monitor",
        "conditions": lambda r: (
            r.get("label") == "normal"
            and r.get("confidence_band") in ("medium", "low")
        ),
        "decision":   "monitor",
        "priority":   "P3",
        "reason":     "Normal label with moderate confidence — light monitoring recommended",
        "actions": [
            "Continue standard monitoring cycle",
            "No immediate action required",
        ],
    },
]

# Priority → routing target
ROUTING_MAP = {
    "P0": "escalation_queue",
    "P1": "escalation_queue",
    "P2": "human_review",
    "P3": "auto_approve",
    "P4": "auto_approve",
}

# Priority → alert trigger
ALERT_PRIORITIES = {"P0", "P1"}


# ── Agent class ───────────────────────────────────────────────────────────────

class DecisionAgent:
    """
    Agent 4 — Decision

    Final agent in the pipeline. Converts classification results into
    concrete decisions with priority levels, recommended actions, and
    routing targets for the confidence router.

    Usage:
        agent = DecisionAgent()
        results = agent.run(records)   # records from ClassificationAgent
    """

    def __init__(self):
        self._results: List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make decisions for a batch of classified records.
        Returns enriched records with decision fields added.
        """
        log.info(f"Making decisions for {len(records)} records")
        self._results = [self._decide(r) for r in records]

        # Stats
        decision_counts = {}
        priority_counts = {}
        alerts          = 0

        for r in self._results:
            d = r["decision"]
            p = r["priority"]
            decision_counts[d] = decision_counts.get(d, 0) + 1
            priority_counts[p] = priority_counts.get(p, 0) + 1
            if r["alert_triggered"]:
                alerts += 1

        log.info(
            f"Decisions complete — "
            + " | ".join(f"{k}={v}" for k, v in sorted(decision_counts.items()))
            + f" | alerts={alerts}"
        )
        return self._results

    def run_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision for a single record."""
        return self._decide(record)

    def summary(self) -> Dict[str, Any]:
        """Batch summary after run()."""
        if not self._results:
            return {"status": "no_results"}

        decision_counts = {}
        priority_counts = {}
        routing_counts  = {}

        for r in self._results:
            d = r["decision"]
            p = r["priority"]
            rt = r["routing_target"]
            decision_counts[d]  = decision_counts.get(d, 0) + 1
            priority_counts[p]  = priority_counts.get(p, 0) + 1
            routing_counts[rt]  = routing_counts.get(rt, 0) + 1

        return {
            "total_records":   len(self._results),
            "decisions":       decision_counts,
            "priorities":      dict(sorted(priority_counts.items())),
            "routing":         routing_counts,
            "alerts_fired":    sum(1 for r in self._results if r["alert_triggered"]),
            "shutdown_events": decision_counts.get("shutdown", 0),
            "escalations":     decision_counts.get("escalate", 0),
            "auto_approved":   routing_counts.get("auto_approve", 0),
            "human_queue":     routing_counts.get("human_review", 0),
            "escalation_queue":routing_counts.get("escalation_queue", 0),
        }

    def save(self, path: str = "results/decision_output.jsonl") -> str:
        """Save decision records to JSONL."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in self._results:
                f.write(json.dumps(r) + "\n")
        log.info(f"Saved {len(self._results)} records → {out_path}")
        return str(out_path)

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Return only records that triggered alerts (P0/P1)."""
        return [r for r in self._results if r["alert_triggered"]]

    def get_escalation_queue(self) -> List[Dict[str, Any]]:
        """Return records routed to escalation queue, sorted by priority."""
        queue = [r for r in self._results if r["routing_target"] == "escalation_queue"]
        return sorted(queue, key=lambda r: r["priority_int"])

    def get_human_review_queue(self) -> List[Dict[str, Any]]:
        """Return records routed to human review, sorted by priority."""
        queue = [r for r in self._results if r["routing_target"] == "human_review"]
        return sorted(queue, key=lambda r: r["priority_int"])

    # ── Core decision logic ───────────────────────────────────────────────────

    def _decide(self, record: Dict[str, Any]) -> Dict[str, Any]:
        r = dict(record)

        # Match first applicable rule
        matched_rule = None
        for rule in DECISION_RULES:
            try:
                if rule["conditions"](r):
                    matched_rule = rule
                    break
            except Exception:
                continue

        # Fallback if no rule matched
        if matched_rule is None:
            matched_rule = {
                "decision": "review",
                "priority": "P2",
                "reason":   "No decision rule matched — defaulting to human review",
                "actions":  ["Manual review required — unhandled classification case"],
            }

        decision  = matched_rule["decision"]
        priority  = matched_rule["priority"]
        priority_int = int(priority[1])  # "P0" → 0
        reason    = matched_rule["reason"]
        actions   = matched_rule["actions"]

        # Alert for P0/P1
        alert_triggered = priority in ALERT_PRIORITIES
        alert_message   = ""
        if alert_triggered:
            severity_str = r.get("max_severity", "unknown").upper()
            alert_message = (
                f"[{priority}] ALERT — {decision.upper()} | "
                f"Record {r.get('record_id', 'unknown')} | "
                f"Severity: {severity_str} | "
                f"{reason}"
            )

        # Routing target
        routing_target = ROUTING_MAP.get(priority, "human_review")

        r["decision"]           = decision
        r["priority"]           = priority
        r["priority_int"]       = priority_int
        r["action_reason"]      = reason
        r["recommended_actions"]= actions
        r["alert_triggered"]    = alert_triggered
        r["alert_message"]      = alert_message
        r["pipeline_complete"]  = True
        r["agent4_status"]      = "decided"
        r["routing_target"]     = routing_target
        r["decided_at"]         = datetime.now().isoformat()

        return r


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 60)
    print("HITL-Ops | Agent 4 — Decision Agent")
    print("=" * 60)

    # ── Run full pipeline ─────────────────────────────────────────────────────
    try:
        from extraction_agent     import ExtractionAgent
        from anomaly_agent        import AnomalyAgent
        from classification_agent import ClassificationAgent

        print("\n[Step 1] Agent 1 — Extraction...")
        extractor = ExtractionAgent(
            mode="pack", profile="wltp", duration_s=3600,
            sample_every=60, initial_soc=0.95,
            inject_faults=True, fault_ratio=0.25,
        )
        records = extractor.run()
        print(f"  {len(records)} records extracted")

        print("[Step 2] Agent 2 — Anomaly Detection...")
        records = AnomalyAgent(suppress_severities=["low"]).run(records)
        print(f"  Anomaly detection done")

        print("[Step 3] Agent 3 — Classification...")
        records = ClassificationAgent().run(records)
        print(f"  Classification done")

    except ImportError as e:
        print(f"\n[Steps 1-3] Agents not found ({e}) — using synthetic records")
        records = [
            {
                "record_id": "syn-001", "timestamp": 0.0,
                "label": "normal", "confidence": 0.85,
                "confidence_band": "high", "conflict": False,
                "anomalies": [], "anomaly_count": 0,
                "max_severity": "none", "severity_score": 0.0,
                "requires_human": False,
            },
            {
                "record_id": "syn-002", "timestamp": 600.0,
                "label": "critical", "confidence": 0.98,
                "confidence_band": "high", "conflict": False,
                "anomalies": [
                    {"rule": "thermal_runaway_risk", "severity": "critical",
                     "feature": "temp_c", "value": 63.0, "threshold": 60.0,
                     "message": "Thermal runaway risk"},
                ],
                "anomaly_count": 1, "max_severity": "critical",
                "severity_score": 1.0, "requires_human": True,
            },
            {
                "record_id": "syn-003", "timestamp": 1200.0,
                "label": "critical", "confidence": 0.82,
                "confidence_band": "high", "conflict": False,
                "anomalies": [
                    {"rule": "deep_discharge", "severity": "critical",
                     "feature": "soc", "value": 0.03, "threshold": 0.05,
                     "message": "Deep discharge"},
                    {"rule": "undervoltage_critical", "severity": "critical",
                     "feature": "v_terminal", "value": 2.91, "threshold": 3.0,
                     "message": "Undervoltage critical"},
                ],
                "anomaly_count": 2, "max_severity": "critical",
                "severity_score": 1.0, "requires_human": True,
            },
            {
                "record_id": "syn-004", "timestamp": 1800.0,
                "label": "warning", "confidence": 0.75,
                "confidence_band": "medium", "conflict": True,
                "anomalies": [
                    {"rule": "overvoltage_warning", "severity": "medium",
                     "feature": "v_terminal", "value": 4.18, "threshold": 4.15,
                     "message": "Overvoltage warning"},
                ],
                "anomaly_count": 1, "max_severity": "medium",
                "severity_score": 0.5, "requires_human": True,
            },
        ]

    # ── Run Agent 4 ───────────────────────────────────────────────────────────
    print("[Step 4] Agent 4 — Decision...")
    decision_agent = DecisionAgent()
    results        = decision_agent.run(records)
    print(f"  Decisions made")

    # ── Print full pipeline output ────────────────────────────────────────────
    print("\n[Results] Full Pipeline Output (Agent 1 → 4):")
    print("-" * 70)

    DECISION_ICON = {
        "ignore":   "⬜",
        "monitor":  "🔵",
        "review":   "🟡",
        "escalate": "🟠",
        "shutdown": "🔴",
    }
    PRIORITY_ICON = {
        "P0": "🚨", "P1": "⚠️ ", "P2": "📋", "P3": "👁 ", "P4": "✅",
    }

    for r in results:
        d_icon = DECISION_ICON.get(r["decision"], "❓")
        p_icon = PRIORITY_ICON.get(r["priority"], "❓")
        print(
            f"  {d_icon} {r['decision'].upper():8s} | "
            f"{p_icon} {r['priority']} | "
            f"label={r['label']:8s} | "
            f"conf={r['confidence']:.2f} | "
            f"→ {r['routing_target']:20s} | "
            f"t={r['timestamp']:.0f}s"
        )
        if r["alert_triggered"]:
            print(f"    🔔 {r['alert_message']}")
        if r["recommended_actions"]:
            for action in r["recommended_actions"][:2]:   # show first 2 actions
                print(f"    • {action}")

    # ── Alerts ────────────────────────────────────────────────────────────────
    alerts = decision_agent.get_alerts()
    if alerts:
        print(f"\n[🔔 ALERTS FIRED: {len(alerts)}]")
        print("-" * 70)
        for r in alerts:
            print(f"  {r['alert_message']}")

    # ── Queues ────────────────────────────────────────────────────────────────
    esc_queue    = decision_agent.get_escalation_queue()
    review_queue = decision_agent.get_human_review_queue()
    print(f"\n[Routing Summary]")
    print(f"  🔴 Escalation queue : {len(esc_queue)} records")
    print(f"  🟡 Human review     : {len(review_queue)} records")
    print(f"  ✅ Auto-approved    : {sum(1 for r in results if r['routing_target'] == 'auto_approve')} records")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[Summary]")
    print("-" * 70)
    summary = decision_agent.summary()
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")

    # Save
    path = decision_agent.save("results/decision_output.jsonl")
    print(f"\n  Saved → {path}")

    print("\n" + "=" * 60)
    print("Agent 4 — DONE. Pipeline complete ✅")
    print("Next: confidence_router.py")
    print("=" * 60)
