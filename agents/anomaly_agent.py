"""
Agent 2 — Anomaly Detection Agent
HITL-Ops: Human-in-the-Loop ML Operations Platform

Role:
    Receives structured feature records from Agent 1 (ExtractionAgent)
    and applies rule-based + statistical anomaly detection grounded in
    real battery ECM physics.

    Detects:
    - Undervoltage / Overvoltage
    - Thermal warning / Thermal runaway risk
    - Deep discharge
    - Cell imbalance (pack mode)
    - Internal resistance growth (aging signal)
    - Rapid voltage drop (dv/dt fault)
    - Rapid temperature rise (dt/dt thermal event)
    - SOC inconsistency

Output per record (adds to Agent 1 dict):
    {
        "anomalies":        list[dict],  # each flagged anomaly
        "anomaly_count":    int,
        "max_severity":     str,         # "none" | "low" | "medium" | "high" | "critical"
        "severity_score":   float,       # 0.0 – 1.0 (used by Agent 3 for classification)
        "anomaly_summary":  str,         # human-readable one-liner
        "agent2_status":    str,         # "ok" | "anomaly_detected" | "error"
    }

Each anomaly dict:
    {
        "rule":        str,    # rule name
        "feature":     str,    # which feature triggered it
        "value":       float,  # actual value
        "threshold":   float,  # threshold that was breached
        "severity":    str,    # "low" | "medium" | "high" | "critical"
        "message":     str,    # human-readable description
    }

Author: Prateek Gaur
Project: hitl-ops
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AnomalyAgent] %(levelname)s — %(message)s",
)
log = logging.getLogger("AnomalyAgent")


# ── Severity ranking ──────────────────────────────────────────────────────────

SEVERITY_RANK = {
    "none":     0,
    "low":      1,
    "medium":   2,
    "high":     3,
    "critical": 4,
}

SEVERITY_SCORE = {
    "none":     0.0,
    "low":      0.25,
    "medium":   0.50,
    "high":     0.75,
    "critical": 1.00,
}


# ── Anomaly Rules ─────────────────────────────────────────────────────────────

# Each rule: (rule_name, feature, condition_fn, threshold, severity, message_template)
# condition_fn(value, threshold) -> bool : True means anomaly detected

ANOMALY_RULES = [
    # ── Voltage rules ────────────────────────────────────────────────────────
    {
        "rule":      "undervoltage_critical",
        "feature":   "v_terminal",
        "condition": lambda v, t: v < t,
        "threshold": 3.0,
        "severity":  "critical",
        "message":   "Terminal voltage {value:.3f}V below critical threshold {threshold}V — cell may be damaged",
    },
    {
        "rule":      "undervoltage_warning",
        "feature":   "v_terminal",
        "condition": lambda v, t: v < t,
        "threshold": 3.2,
        "severity":  "medium",
        "message":   "Terminal voltage {value:.3f}V approaching undervoltage limit {threshold}V",
    },
    {
        "rule":      "overvoltage_critical",
        "feature":   "v_terminal",
        "condition": lambda v, t: v > t,
        "threshold": 4.25,
        "severity":  "critical",
        "message":   "Terminal voltage {value:.3f}V above critical threshold {threshold}V — overcharge risk",
    },
    {
        "rule":      "overvoltage_warning",
        "feature":   "v_terminal",
        "condition": lambda v, t: v > t,
        "threshold": 4.15,
        "severity":  "medium",
        "message":   "Terminal voltage {value:.3f}V approaching overvoltage limit {threshold}V",
    },

    # ── Thermal rules ─────────────────────────────────────────────────────────
    {
        "rule":      "thermal_runaway_risk",
        "feature":   "temp_c",
        "condition": lambda v, t: v > t,
        "threshold": 60.0,
        "severity":  "critical",
        "message":   "Temperature {value:.1f}°C exceeds {threshold}°C — thermal runaway risk",
    },
    {
        "rule":      "thermal_warning",
        "feature":   "temp_c",
        "condition": lambda v, t: v > t,
        "threshold": 45.0,
        "severity":  "high",
        "message":   "Temperature {value:.1f}°C above warning threshold {threshold}°C",
    },
    {
        "rule":      "thermal_elevated",
        "feature":   "temp_c",
        "condition": lambda v, t: v > t,
        "threshold": 35.0,
        "severity":  "low",
        "message":   "Temperature {value:.1f}°C elevated above nominal {threshold}°C",
    },
    {
        "rule":      "rapid_temp_rise",
        "feature":   "dt_dt",
        "condition": lambda v, t: v > t,
        "threshold": 0.05,   # °C per second
        "severity":  "high",
        "message":   "Rapid temperature rise {value:.4f}°C/s — possible thermal event",
    },

    # ── SOC rules ─────────────────────────────────────────────────────────────
    {
        "rule":      "deep_discharge",
        "feature":   "soc",
        "condition": lambda v, t: v < t,
        "threshold": 0.05,
        "severity":  "critical",
        "message":   "SOC {value:.3f} below deep discharge threshold {threshold} — cell damage risk",
    },
    {
        "rule":      "low_soc_warning",
        "feature":   "soc",
        "condition": lambda v, t: v < t,
        "threshold": 0.10,
        "severity":  "medium",
        "message":   "SOC {value:.3f} approaching low threshold {threshold}",
    },

    # ── Resistance / aging rules ──────────────────────────────────────────────
    {
        "rule":      "resistance_growth_high",
        "feature":   "r0_eff",
        "condition": lambda v, t: v > t,
        "threshold": 0.005,
        "severity":  "high",
        "message":   "Internal resistance {value:.5f}Ω significantly above nominal — advanced aging",
    },
    {
        "rule":      "resistance_growth_warn",
        "feature":   "r0_eff",
        "condition": lambda v, t: v > t,
        "threshold": 0.004,
        "severity":  "medium",
        "message":   "Internal resistance {value:.5f}Ω above nominal {threshold}Ω — aging signal",
    },

    # ── Rate of change rules ──────────────────────────────────────────────────
    {
        "rule":      "rapid_voltage_drop",
        "feature":   "dv_dt",
        "condition": lambda v, t: v < t,
        "threshold": -0.005,  # V per second
        "severity":  "high",
        "message":   "Rapid voltage drop {value:.5f}V/s — possible fault or load spike",
    },

    # ── Voltage drop (OCV - V_terminal) ──────────────────────────────────────
    {
        "rule":      "high_voltage_drop",
        "feature":   "v_drop",
        "condition": lambda v, t: v > t,
        "threshold": 0.15,
        "severity":  "medium",
        "message":   "Voltage drop {value:.4f}V above threshold {threshold}V — high internal loss",
    },
]

# Pack-only rules (only apply when pack fields are present)
PACK_ANOMALY_RULES = [
    {
        "rule":      "cell_imbalance_critical",
        "feature":   "imbalance",
        "condition": lambda v, t: v > t,
        "threshold": 0.08,
        "severity":  "critical",
        "message":   "Cell SOC imbalance σ={value:.4f} exceeds critical threshold {threshold} — balancing failure",
    },
    {
        "rule":      "cell_imbalance_warning",
        "feature":   "imbalance",
        "condition": lambda v, t: v > t,
        "threshold": 0.05,
        "severity":  "medium",
        "message":   "Cell SOC imbalance σ={value:.4f} above warning threshold {threshold}",
    },
    {
        "rule":      "pack_temp_critical",
        "feature":   "temp_max_c",
        "condition": lambda v, t: v > t,
        "threshold": 60.0,
        "severity":  "critical",
        "message":   "Pack max temperature {value:.1f}°C exceeds {threshold}°C — thermal runaway risk",
    },
    {
        "rule":      "pack_soc_spread",
        "feature":   "pack_soc_min",
        "condition": lambda v, t: v < t,
        "threshold": 0.05,
        "severity":  "critical",
        "message":   "Weakest cell SOC {value:.3f} below deep discharge threshold {threshold}",
    },
]


# ── Agent class ───────────────────────────────────────────────────────────────

class AnomalyAgent:
    """
    Agent 2 — Anomaly Detection

    Applies physics-grounded rule-based detection on battery ECM features.
    Designed to work on single records or full batches from Agent 1.

    Usage:
        agent = AnomalyAgent()
        results = agent.run(records)   # records from ExtractionAgent
    """

    def __init__(
        self,
        custom_rules:       Optional[List[Dict]] = None,
        suppress_severities: Optional[List[str]] = None,
    ):
        """
        Args:
            custom_rules:        Additional rules to merge with defaults
            suppress_severities: List of severity levels to ignore e.g. ["low"]
        """
        self.rules       = ANOMALY_RULES + (custom_rules or [])
        self.pack_rules  = PACK_ANOMALY_RULES
        self.suppress    = set(suppress_severities or [])
        self._results:   List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run anomaly detection on a batch of records from Agent 1.
        Returns enriched records with anomaly fields added.
        """
        log.info(f"Running anomaly detection on {len(records)} records")

        self._results = [self._analyze_record(r) for r in records]

        # Summary stats
        total_anomalies  = sum(r["anomaly_count"] for r in self._results)
        critical_records = sum(1 for r in self._results if r["max_severity"] == "critical")
        clean_records    = sum(1 for r in self._results if r["max_severity"] == "none")

        log.info(
            f"Detection complete — "
            f"{clean_records} clean | "
            f"{total_anomalies} total anomalies | "
            f"{critical_records} critical records"
        )

        return self._results

    def run_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Run anomaly detection on a single record."""
        return self._analyze_record(record)

    def summary(self) -> Dict[str, Any]:
        """Batch summary after run()."""
        if not self._results:
            return {"status": "no_results"}

        severity_counts = {s: 0 for s in SEVERITY_RANK}
        rule_hits       = {}

        for r in self._results:
            severity_counts[r["max_severity"]] += 1
            for a in r["anomalies"]:
                rule_hits[a["rule"]] = rule_hits.get(a["rule"], 0) + 1

        return {
            "total_records":    len(self._results),
            "severity_counts":  severity_counts,
            "top_rules":        sorted(rule_hits.items(), key=lambda x: -x[1])[:5],
            "critical_rate":    round(severity_counts["critical"] / len(self._results), 3),
            "clean_rate":       round(severity_counts["none"]     / len(self._results), 3),
        }

    def save(self, path: str = "results/anomaly_output.jsonl") -> str:
        """Save anomaly-enriched records to JSONL."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in self._results:
                f.write(json.dumps(r) + "\n")
        log.info(f"Saved {len(self._results)} records → {out_path}")
        return str(out_path)

    # ── Core detection ────────────────────────────────────────────────────────

    def _analyze_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all rules to one record and return enriched dict."""
        r        = dict(record)
        anomalies = []

        # Cell-level rules
        for rule in self.rules:
            anomaly = self._apply_rule(r, rule)
            if anomaly:
                anomalies.append(anomaly)

        # Pack-level rules (only if pack data present)
        if r.get("imbalance") is not None:
            for rule in self.pack_rules:
                anomaly = self._apply_rule(r, rule)
                if anomaly:
                    anomalies.append(anomaly)

        # Filter suppressed severities
        anomalies = [a for a in anomalies if a["severity"] not in self.suppress]

        # Deduplicate: if both warning and critical fire on same feature,
        # keep only the highest severity
        anomalies = self._deduplicate(anomalies)

        # Compute max severity and score
        max_severity = "none"
        for a in anomalies:
            if SEVERITY_RANK[a["severity"]] > SEVERITY_RANK[max_severity]:
                max_severity = a["severity"]

        severity_score = SEVERITY_SCORE[max_severity]

        # Human-readable summary
        if not anomalies:
            summary = "No anomalies detected — all parameters within normal range"
        elif len(anomalies) == 1:
            summary = anomalies[0]["message"]
        else:
            top = max(anomalies, key=lambda a: SEVERITY_RANK[a["severity"]])
            summary = f"{top['message']} (+ {len(anomalies)-1} other issue(s))"

        r["anomalies"]       = anomalies
        r["anomaly_count"]   = len(anomalies)
        r["max_severity"]    = max_severity
        r["severity_score"]  = severity_score
        r["anomaly_summary"] = summary
        r["agent2_status"]   = "anomaly_detected" if anomalies else "ok"

        return r

    def _apply_rule(self, record: Dict, rule: Dict) -> Optional[Dict]:
        """Apply a single rule to a record. Returns anomaly dict or None."""
        feature   = rule["feature"]
        value     = record.get(feature)

        # Skip if feature not present or None
        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        threshold = rule["threshold"]

        if rule["condition"](value, threshold):
            return {
                "rule":      rule["rule"],
                "feature":   feature,
                "value":     round(value, 6),
                "threshold": threshold,
                "severity":  rule["severity"],
                "message":   rule["message"].format(value=value, threshold=threshold),
            }
        return None

    def _deduplicate(self, anomalies: List[Dict]) -> List[Dict]:
        """
        If multiple rules fire on the same feature, keep only the
        highest severity one to avoid redundant alerts.
        """
        best: Dict[str, Dict] = {}
        for a in anomalies:
            feat = a["feature"]
            if feat not in best:
                best[feat] = a
            elif SEVERITY_RANK[a["severity"]] > SEVERITY_RANK[best[feat]["severity"]]:
                best[feat] = a
        return list(best.values())


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Import Agent 1
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 60)
    print("HITL-Ops | Agent 2 — Anomaly Detection Agent")
    print("=" * 60)

    # ── Try to import ExtractionAgent ─────────────────────────────────────────
    try:
        from extraction_agent import ExtractionAgent

        print("\n[Step 1] Running Agent 1 — Extraction...")
        extractor = ExtractionAgent(
            mode="pack",
            profile="wltp",
            duration_s=3600,
            sample_every=60,
            initial_soc=0.95,
            inject_faults=True,
            fault_ratio=0.25,
        )
        records = extractor.run()
        print(f"  Extracted {len(records)} records")

    except ImportError:
        print("\n[Step 1] ExtractionAgent not found — using synthetic test records")
        # Synthetic records to test anomaly rules independently
        records = [
            # Normal record
            {
                "record_id": "test-001", "timestamp": 0.0, "source": "test",
                "current_a": 25.0, "soc": 0.85, "v_terminal": 3.85,
                "v_ocv": 3.87, "v_rc1": 0.001, "v_rc2": 0.0008,
                "temp_c": 27.0, "r0_eff": 0.0021,
                "pack_voltage_v": 53.9, "pack_soc_mean": 0.85,
                "pack_soc_min": 0.83, "pack_soc_max": 0.87,
                "imbalance": 0.01, "temp_max_c": 28.0, "temp_mean_c": 27.0,
                "v_drop": 0.02, "dv_dt": -0.0001, "dt_dt": 0.001,
                "soc_change_rate": -0.00003, "extraction_status": "ok", "notes": [],
            },
            # Thermal warning
            {
                "record_id": "test-002", "timestamp": 60.0, "source": "test",
                "current_a": 25.0, "soc": 0.60, "v_terminal": 3.76,
                "v_ocv": 3.78, "v_rc1": 0.001, "v_rc2": 0.0008,
                "temp_c": 48.0, "r0_eff": 0.0022,
                "pack_voltage_v": 52.6, "pack_soc_mean": 0.60,
                "pack_soc_min": 0.57, "pack_soc_max": 0.63,
                "imbalance": 0.03, "temp_max_c": 51.0, "temp_mean_c": 48.0,
                "v_drop": 0.02, "dv_dt": -0.0002, "dt_dt": 0.06,
                "soc_change_rate": -0.00003, "extraction_status": "ok", "notes": [],
            },
            # Critical — undervoltage + deep discharge + imbalance
            {
                "record_id": "test-003", "timestamp": 120.0, "source": "test",
                "current_a": 25.0, "soc": 0.04, "v_terminal": 2.95,
                "v_ocv": 3.10, "v_rc1": 0.001, "v_rc2": 0.0008,
                "temp_c": 55.0, "r0_eff": 0.0055,
                "pack_voltage_v": 41.3, "pack_soc_mean": 0.04,
                "pack_soc_min": 0.02, "pack_soc_max": 0.08,
                "imbalance": 0.09, "temp_max_c": 62.0, "temp_mean_c": 55.0,
                "v_drop": 0.15, "dv_dt": -0.008, "dt_dt": 0.07,
                "soc_change_rate": -0.00005, "extraction_status": "warn",
                "notes": ["SOC out of range"],
            },
        ]

    # ── Run Agent 2 ───────────────────────────────────────────────────────────
    print("\n[Step 2] Running Agent 2 — Anomaly Detection...")
    agent = AnomalyAgent(suppress_severities=["low"])
    results = agent.run(records)

    # ── Print results — anomalous records only ────────────────────────────────
    anomalous = [r for r in results if r["agent2_status"] == "anomaly_detected"]
    clean     = [r for r in results if r["agent2_status"] == "ok"]

    print(f"\n[Results] {len(anomalous)} anomalous / {len(clean)} clean / {len(results)} total")
    print("-" * 60)

    for r in anomalous:
        print(f"\n  Record  : {r['record_id']}")
        print(f"  Time    : {r['timestamp']:.0f}s")
        print(f"  Severity: {r['max_severity'].upper()} (score={r['severity_score']:.2f})")
        print(f"  Summary : {r['anomaly_summary']}")
        if r["anomalies"]:
            for a in r["anomalies"]:
                print(f"    [{a['severity'].upper():8s}] {a['rule']:35s} — {a['message']}")

    # ── Batch summary ─────────────────────────────────────────────────────────
    print("\n[Summary]")
    print("-" * 60)
    summary = agent.summary()
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    # Save output
    path = agent.save("results/anomaly_output.jsonl")
    print(f"\n  Saved → {path}")

    print("\n" + "=" * 60)
    print("Agent 2 — DONE. Passing to Agent 3.")
    print("=" * 60)
