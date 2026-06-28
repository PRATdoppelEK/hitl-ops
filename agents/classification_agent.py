"""
Agent 3 — Classification Agent
HITL-Ops: Human-in-the-Loop ML Operations Platform

Role:
    Receives anomaly-enriched records from Agent 2 and classifies
    each into a final label with a confidence score.

    Classification labels:
        - "normal"   → no issues, system healthy
        - "warning"  → anomaly detected, monitor closely
        - "critical" → immediate action required

    Confidence score (0.0 – 1.0):
        - High   >= 0.75 → confident classification
        - Medium  0.45–0.74 → uncertain, may need human review
        - Low    < 0.45  → very uncertain, escalate to human

    Strategy:
        - Rule-based classifier (primary) — deterministic, physics-grounded
        - Confidence estimator — based on severity score + anomaly count + consistency
        - Conflict detector — flags records where rules disagree

Output per record (adds to Agent 2 dict):
    {
        "label":              str,    # "normal" | "warning" | "critical"
        "confidence":         float,  # 0.0 – 1.0
        "confidence_band":    str,    # "high" | "medium" | "low"
        "label_reason":       str,    # why this label was assigned
        "conflict":           bool,   # True if rules disagreed
        "conflict_notes":     list,   # details of any conflict
        "agent3_status":      str,    # "classified" | "conflict" | "error"
        "requires_human":     bool,   # True if confidence is medium/low
    }

Author: Prateek Gaur
Project: hitl-ops
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ClassificationAgent] %(levelname)s — %(message)s",
)
log = logging.getLogger("ClassificationAgent")


# ── Classification thresholds ─────────────────────────────────────────────────

# Severity score → label mapping (from Agent 2)
# severity_score is 0.0 (none) → 1.0 (critical)
SEVERITY_TO_LABEL = [
    (0.00, 0.10, "normal"),    # score 0.0 – 0.10
    (0.10, 0.60, "warning"),   # score 0.10 – 0.60
    (0.60, 1.01, "critical"),  # score 0.60 – 1.0
]

# Confidence band thresholds
CONFIDENCE_HIGH   = 0.75
CONFIDENCE_MEDIUM = 0.45

# Anomaly rules that always → critical regardless of score
HARD_CRITICAL_RULES = {
    "undervoltage_critical",
    "overvoltage_critical",
    "thermal_runaway_risk",
    "deep_discharge",
    "cell_imbalance_critical",
    "pack_temp_critical",
    "pack_soc_spread",
}

# Anomaly rules that always → warning (at minimum)
HARD_WARNING_RULES = {
    "undervoltage_warning",
    "overvoltage_warning",
    "thermal_warning",
    "resistance_growth_high",
    "resistance_growth_warn",
    "rapid_voltage_drop",
    "rapid_temp_rise",
    "high_voltage_drop",
    "low_soc_warning",
    "cell_imbalance_warning",
}


# ── Agent class ───────────────────────────────────────────────────────────────

class ClassificationAgent:
    """
    Agent 3 — Classification

    Takes anomaly-enriched records from Agent 2 and assigns:
    - A label: normal / warning / critical
    - A confidence score: how certain we are
    - A human review flag: when confidence is too low to trust

    Usage:
        agent = ClassificationAgent()
        results = agent.run(records)   # records from AnomalyAgent
    """

    def __init__(
        self,
        confidence_high:   float = CONFIDENCE_HIGH,
        confidence_medium: float = CONFIDENCE_MEDIUM,
    ):
        self.conf_high   = confidence_high
        self.conf_medium = confidence_medium
        self._results: List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a batch of records from Agent 2.
        Returns enriched records with classification fields added.
        """
        log.info(f"Classifying {len(records)} records")
        self._results = [self._classify_record(r) for r in records]

        # Stats
        label_counts = {"normal": 0, "warning": 0, "critical": 0}
        band_counts  = {"high": 0, "medium": 0, "low": 0}
        human_needed = 0

        for r in self._results:
            label_counts[r["label"]] += 1
            band_counts[r["confidence_band"]] += 1
            if r["requires_human"]:
                human_needed += 1

        log.info(
            f"Classification complete — "
            f"normal={label_counts['normal']} | "
            f"warning={label_counts['warning']} | "
            f"critical={label_counts['critical']} | "
            f"human_needed={human_needed}"
        )
        return self._results

    def run_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single record."""
        return self._classify_record(record)

    def summary(self) -> Dict[str, Any]:
        """Batch summary after run()."""
        if not self._results:
            return {"status": "no_results"}

        label_counts = {"normal": 0, "warning": 0, "critical": 0}
        band_counts  = {"high": 0, "medium": 0, "low": 0}
        conflicts    = 0
        confidences  = []

        for r in self._results:
            label_counts[r["label"]] += 1
            band_counts[r["confidence_band"]] += 1
            confidences.append(r["confidence"])
            if r["conflict"]:
                conflicts += 1

        return {
            "total_records":    len(self._results),
            "label_counts":     label_counts,
            "confidence_bands": band_counts,
            "mean_confidence":  round(float(np.mean(confidences)), 3),
            "conflicts":        conflicts,
            "requires_human":   sum(1 for r in self._results if r["requires_human"]),
            "auto_approvable":  sum(1 for r in self._results if not r["requires_human"]),
        }

    def save(self, path: str = "results/classification_output.jsonl") -> str:
        """Save classified records to JSONL."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in self._results:
                f.write(json.dumps(r) + "\n")
        log.info(f"Saved {len(self._results)} records → {out_path}")
        return str(out_path)

    # ── Core classification ───────────────────────────────────────────────────

    def _classify_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        r = dict(record)

        anomalies      = r.get("anomalies", [])
        severity_score = r.get("severity_score", 0.0)
        max_severity   = r.get("max_severity", "none")
        anomaly_count  = r.get("anomaly_count", 0)

        # Step 1 — primary label from severity score
        primary_label  = self._label_from_severity(severity_score)

        # Step 2 — hard rule overrides
        rule_names     = {a["rule"] for a in anomalies}
        override_label, override_reason = self._apply_hard_rules(rule_names, primary_label)

        # Step 3 — conflict detection
        conflict, conflict_notes = self._detect_conflict(
            primary_label, override_label, anomalies, severity_score
        )

        final_label  = override_label
        label_reason = override_reason

        # Step 4 — confidence estimation
        confidence   = self._estimate_confidence(
            final_label, severity_score, anomaly_count,
            rule_names, conflict, max_severity
        )

        # Step 5 — confidence band
        if confidence >= self.conf_high:
            band = "high"
        elif confidence >= self.conf_medium:
            band = "medium"
        else:
            band = "low"

        # Step 6 — human review flag
        # Humans review: medium/low confidence OR conflict OR critical
        requires_human = (
            band in ("medium", "low")
            or conflict
            or final_label == "critical"
        )

        r["label"]           = final_label
        r["confidence"]      = round(confidence, 4)
        r["confidence_band"] = band
        r["label_reason"]    = label_reason
        r["conflict"]        = conflict
        r["conflict_notes"]  = conflict_notes
        r["agent3_status"]   = "conflict" if conflict else "classified"
        r["requires_human"]  = requires_human

        return r

    def _label_from_severity(self, score: float) -> str:
        """Map severity score → label."""
        for low, high, label in SEVERITY_TO_LABEL:
            if low <= score < high:
                return label
        return "critical"

    def _apply_hard_rules(
        self, rule_names: set, primary_label: str
    ) -> Tuple[str, str]:
        """
        Override label if hard-critical or hard-warning rules fired.
        Returns (final_label, reason).
        """
        triggered_critical = rule_names & HARD_CRITICAL_RULES
        triggered_warning  = rule_names & HARD_WARNING_RULES

        if triggered_critical:
            rules_str = ", ".join(sorted(triggered_critical))
            return "critical", f"Hard critical rule(s) triggered: {rules_str}"

        if triggered_warning and primary_label == "normal":
            rules_str = ", ".join(sorted(triggered_warning))
            return "warning", f"Hard warning rule(s) triggered: {rules_str}"

        if primary_label == "normal" and not triggered_critical and not triggered_warning:
            return "normal", "No anomaly rules triggered — all parameters nominal"

        return primary_label, f"Severity score {primary_label} classification"

    def _detect_conflict(
        self,
        primary_label: str,
        override_label: str,
        anomalies: List[Dict],
        severity_score: float,
    ) -> Tuple[bool, List[str]]:
        """
        Detect conflicts — cases where rules disagree with each other
        or with the severity score.
        """
        notes   = []
        conflict = False

        # Conflict 1: severity says one thing, hard rules say another
        if primary_label != override_label:
            notes.append(
                f"Severity score suggested '{primary_label}' "
                f"but hard rules overrode to '{override_label}'"
            )
            conflict = True

        # Conflict 2: mix of critical and low-severity anomalies
        if anomalies:
            severities = {a["severity"] for a in anomalies}
            if "critical" in severities and "medium" in severities and len(anomalies) >= 3:
                notes.append(
                    "Mixed severity anomalies detected — "
                    "critical and medium rules firing together"
                )
                conflict = True

        # Conflict 3: high severity score but few anomalies (unusual)
        if severity_score >= 0.75 and len(anomalies) == 1:
            notes.append(
                f"High severity score ({severity_score:.2f}) "
                f"from only 1 anomaly — confidence reduced"
            )
            conflict = True

        return conflict, notes

    def _estimate_confidence(
        self,
        label: str,
        severity_score: float,
        anomaly_count: int,
        rule_names: set,
        conflict: bool,
        max_severity: str,
    ) -> float:
        """
        Estimate confidence in the classification.

        Higher confidence when:
        - Severity score is extreme (clearly normal or clearly critical)
        - Hard rules fired (deterministic)
        - Multiple consistent anomalies
        - No conflict

        Lower confidence when:
        - Severity score is borderline (near thresholds)
        - No hard rules fired
        - Conflict detected
        - Only one anomaly
        """
        base = 0.5

        # Distance from decision boundaries (0.1 and 0.6)
        dist_to_boundary = min(
            abs(severity_score - 0.10),
            abs(severity_score - 0.60),
        )
        boundary_boost = min(dist_to_boundary * 1.5, 0.30)
        base += boundary_boost

        # Hard rules fired → more confident
        hard_critical_hit = bool(rule_names & HARD_CRITICAL_RULES)
        hard_warning_hit  = bool(rule_names & HARD_WARNING_RULES)
        if hard_critical_hit:
            base += 0.20
        elif hard_warning_hit:
            base += 0.10

        # Multiple consistent anomalies → more confident
        if anomaly_count == 0:
            base += 0.10   # clean record is confident
        elif anomaly_count >= 3:
            base += 0.10   # many consistent flags
        elif anomaly_count == 1:
            base -= 0.05   # single anomaly — less certain

        # Conflict → less confident
        if conflict:
            base -= 0.20

        # Normal label with zero anomalies → very confident
        if label == "normal" and anomaly_count == 0:
            base = max(base, 0.85)

        # Critical with multiple hard rules → very confident
        if label == "critical" and hard_critical_hit and anomaly_count >= 2:
            base = max(base, 0.88)

        return float(np.clip(base, 0.05, 0.98))


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 60)
    print("HITL-Ops | Agent 3 — Classification Agent")
    print("=" * 60)

    # ── Run Agent 1 + 2 first ─────────────────────────────────────────────────
    try:
        from extraction_agent import ExtractionAgent
        from anomaly_agent    import AnomalyAgent

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

        print("\n[Step 2] Running Agent 2 — Anomaly Detection...")
        anomaly_agent = AnomalyAgent(suppress_severities=["low"])
        records = anomaly_agent.run(records)
        print(f"  Anomaly detection done")

    except ImportError as e:
        print(f"\n[Step 1+2] Agents not found ({e}) — using synthetic records")
        records = [
            {
                "record_id": "syn-001", "timestamp": 0.0,
                "severity_score": 0.0, "max_severity": "none",
                "anomaly_count": 0, "anomalies": [],
                "agent2_status": "ok",
            },
            {
                "record_id": "syn-002", "timestamp": 60.0,
                "severity_score": 0.50, "max_severity": "medium",
                "anomaly_count": 1,
                "anomalies": [{"rule": "overvoltage_warning", "severity": "medium",
                               "feature": "v_terminal", "value": 4.18,
                               "threshold": 4.15, "message": "Overvoltage warning"}],
                "agent2_status": "anomaly_detected",
            },
            {
                "record_id": "syn-003", "timestamp": 120.0,
                "severity_score": 1.0, "max_severity": "critical",
                "anomaly_count": 3,
                "anomalies": [
                    {"rule": "thermal_runaway_risk", "severity": "critical",
                     "feature": "temp_c", "value": 63.0,
                     "threshold": 60.0, "message": "Thermal runaway risk"},
                    {"rule": "pack_temp_critical", "severity": "critical",
                     "feature": "temp_max_c", "value": 67.0,
                     "threshold": 60.0, "message": "Pack temp critical"},
                    {"rule": "rapid_temp_rise", "severity": "high",
                     "feature": "dt_dt", "value": 0.12,
                     "threshold": 0.05, "message": "Rapid temp rise"},
                ],
                "agent2_status": "anomaly_detected",
            },
        ]

    # ── Run Agent 3 ───────────────────────────────────────────────────────────
    print("\n[Step 3] Running Agent 3 — Classification...")
    classifier = ClassificationAgent()
    results    = classifier.run(records)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n[Results] Classification Output:")
    print("-" * 60)

    for r in results:
        band_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(r["confidence_band"], "⚪")
        label_icon = {"normal": "✅", "warning": "⚠️ ", "critical": "🚨"}.get(r["label"], "❓")
        human_flag = "👤 HUMAN" if r["requires_human"] else "🤖 AUTO "

        print(
            f"  {label_icon} {r['label'].upper():8s} | "
            f"{band_icon} conf={r['confidence']:.2f} | "
            f"{human_flag} | "
            f"t={r['timestamp']:.0f}s | "
            f"{r['record_id']}"
        )
        if r["conflict"]:
            for note in r["conflict_notes"]:
                print(f"    ⚡ CONFLICT: {note}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[Summary]")
    print("-" * 60)
    summary = classifier.summary()
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")

    # Save
    path = classifier.save("results/classification_output.jsonl")
    print(f"\n  Saved → {path}")

    print("\n" + "=" * 60)
    print("Agent 3 — DONE. Passing to Agent 4.")
    print("=" * 60)
