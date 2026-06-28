"""
Confidence Router
HITL-Ops: Human-in-the-Loop ML Operations Platform

Role:
    Bridge between the 4-agent pipeline and the human review interface.
    Takes Agent 4's decisions and officially routes each record to the
    correct queue based on confidence, priority, and configurable thresholds.

    Three routing destinations:
        auto_approve      → high confidence, normal/monitor — no human needed
        human_review      → medium confidence or warning — human reviews
        escalation_queue  → critical/shutdown — immediate human attention

    The router also:
        - Applies configurable confidence thresholds (can be tuned)
        - Detects drift: if auto-approve rate drops below threshold → alert
        - Produces a routing manifest (summary of all decisions)
        - Emits a flat queue-ready list for the review interface

Author: Prateek Gaur
Project: hitl-ops
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ConfidenceRouter] %(levelname)s — %(message)s",
)
log = logging.getLogger("ConfidenceRouter")


# ── Routing config ────────────────────────────────────────────────────────────

@dataclass
class RouterConfig:
    """
    Configurable thresholds for the confidence router.
    Tune these without touching agent code.
    """
    # Confidence thresholds
    auto_approve_min_confidence:  float = 0.80   # below this → human_review even if normal
    escalation_min_confidence:    float = 0.45   # below this + critical → review not escalate

    # Priority overrides
    always_escalate_priorities:   List[str] = field(default_factory=lambda: ["P0", "P1"])
    always_review_priorities:     List[str] = field(default_factory=lambda: ["P2"])
    always_auto_priorities:       List[str] = field(default_factory=lambda: ["P3", "P4"])

    # Drift detection
    auto_approve_floor:           float = 0.50   # alert if auto-approve rate drops below this
    escalation_ceiling:           float = 0.40   # alert if escalation rate exceeds this

    # Batch metadata
    batch_id:                     str   = ""
    source:                       str   = "hitl-ops-pipeline"


@dataclass
class RoutedRecord:
    """A record after routing — slim version passed to review interface."""
    record_id:          str
    timestamp:          float
    routing_target:     str       # "auto_approve" | "human_review" | "escalation_queue"
    priority:           str       # P0–P4
    priority_int:       int
    decision:           str       # ignore | monitor | review | escalate | shutdown
    label:              str       # normal | warning | critical
    confidence:         float
    confidence_band:    str
    max_severity:       str
    severity_score:     float
    anomaly_summary:    str
    anomalies:          List[Dict]
    recommended_actions: List[str]
    alert_triggered:    bool
    alert_message:      str
    conflict:           bool
    source:             str
    routed_at:          str
    # Raw sensor values (for display in review interface)
    v_terminal:         Optional[float] = None
    soc:                Optional[float] = None
    temp_c:             Optional[float] = None
    r0_eff:             Optional[float] = None
    imbalance:          Optional[float] = None
    pack_soc_mean:      Optional[float] = None
    temp_max_c:         Optional[float] = None
    # Override flag: router changed Agent 4's routing
    routing_overridden: bool = False
    override_reason:    str  = ""


# ── Router class ──────────────────────────────────────────────────────────────

class ConfidenceRouter:
    """
    Confidence Router

    Takes the full pipeline output (Agent 4 records) and:
    1. Applies threshold-based routing rules
    2. Overrides Agent 4 routing when confidence doesn't match priority
    3. Detects batch-level drift / anomalies in routing distribution
    4. Produces three sorted queues ready for downstream consumption

    Usage:
        router = ConfidenceRouter(config=RouterConfig())
        queues = router.route(records)
        router.save_queues()
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        if not self.config.batch_id:
            self.config.batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")

        self._auto_queue:      List[RoutedRecord] = []
        self._review_queue:    List[RoutedRecord] = []
        self._escalation_queue:List[RoutedRecord] = []
        self._manifest:        Dict[str, Any]     = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def route(self, records: List[Dict[str, Any]]) -> Dict[str, List[RoutedRecord]]:
        """
        Route all records from the pipeline.
        Returns dict with three queues.
        """
        log.info(f"Routing {len(records)} records | batch={self.config.batch_id}")

        # Suppress consecutive duplicates — same top rule firing repeatedly
        records = self._suppress_consecutive_duplicates(records)
        log.info(f"After duplicate suppression: {len(records)} records")

        self._auto_queue       = []
        self._review_queue     = []
        self._escalation_queue = []

        overrides = 0
        for record in records:
            routed, overridden = self._route_record(record)
            if overridden:
                overrides += 1

            if routed.routing_target == "auto_approve":
                self._auto_queue.append(routed)
            elif routed.routing_target == "human_review":
                self._review_queue.append(routed)
            else:
                self._escalation_queue.append(routed)

        # Sort escalation by priority (P0 first)
        self._escalation_queue.sort(key=lambda r: r.priority_int)
        self._review_queue.sort(key=lambda r: r.priority_int)

        # Build manifest
        self._manifest = self._build_manifest(records, overrides)

        # Drift detection
        self._check_drift()

        log.info(
            f"Routing complete — "
            f"auto={len(self._auto_queue)} | "
            f"review={len(self._review_queue)} | "
            f"escalation={len(self._escalation_queue)} | "
            f"overrides={overrides}"
        )

        return {
            "auto_approve":      self._auto_queue,
            "human_review":      self._review_queue,
            "escalation_queue":  self._escalation_queue,
        }

    def print_queues(self):
        """Pretty-print all three queues to console."""
        print("\n" + "=" * 65)
        print(f"  CONFIDENCE ROUTER — {self.config.batch_id}")
        print("=" * 65)

        # Escalation queue
        print(f"\n🔴 ESCALATION QUEUE ({len(self._escalation_queue)} records)")
        print("-" * 65)
        if self._escalation_queue:
            for r in self._escalation_queue:
                override_tag = " [OVERRIDE]" if r.routing_overridden else ""
                print(
                    f"  {r.priority} | {r.decision.upper():8s} | "
                    f"conf={r.confidence:.2f} | "
                    f"t={r.timestamp:.0f}s | "
                    f"{r.record_id}{override_tag}"
                )
                print(f"       ↳ {r.anomaly_summary}")
                if r.alert_message:
                    print(f"       🔔 {r.alert_message}")
        else:
            print("  (empty)")

        # Human review queue
        print(f"\n🟡 HUMAN REVIEW QUEUE ({len(self._review_queue)} records)")
        print("-" * 65)
        if self._review_queue:
            for r in self._review_queue:
                override_tag = " [OVERRIDE]" if r.routing_overridden else ""
                print(
                    f"  {r.priority} | {r.decision.upper():8s} | "
                    f"conf={r.confidence:.2f} | "
                    f"t={r.timestamp:.0f}s | "
                    f"{r.record_id}{override_tag}"
                )
                print(f"       ↳ {r.anomaly_summary}")
        else:
            print("  (empty)")

        # Auto-approve queue
        print(f"\n✅ AUTO-APPROVE QUEUE ({len(self._auto_queue)} records)")
        print("-" * 65)
        label_counts = {}
        for r in self._auto_queue:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1
        for label, count in sorted(label_counts.items()):
            print(f"  {label:10s} → {count} records auto-approved")

        # Manifest summary
        print(f"\n📋 ROUTING MANIFEST")
        print("-" * 65)
        m = self._manifest
        print(f"  Batch ID          : {m['batch_id']}")
        print(f"  Total records     : {m['total_records']}")
        print(f"  Auto-approve rate : {m['auto_approve_rate']:.1%}")
        print(f"  Escalation rate   : {m['escalation_rate']:.1%}")
        print(f"  Routing overrides : {m['routing_overrides']}")
        print(f"  Drift alerts      : {m['drift_alerts']}")
        print(f"  Routed at         : {m['routed_at']}")
        print("=" * 65)

    def save_queues(self, output_dir: str = "results") -> Dict[str, str]:
        """Save all three queues + manifest to JSONL/JSON files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths = {}

        for queue_name, queue in [
            ("escalation_queue", self._escalation_queue),
            ("human_review",     self._review_queue),
            ("auto_approve",     self._auto_queue),
        ]:
            path = out / f"router_{queue_name}.jsonl"
            with open(path, "w") as f:
                for r in queue:
                    f.write(json.dumps(asdict(r)) + "\n")
            paths[queue_name] = str(path)
            log.info(f"Saved {len(queue)} records → {path}")

        manifest_path = out / "router_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)
        paths["manifest"] = str(manifest_path)
        log.info(f"Saved manifest → {manifest_path}")

        return paths

    def get_escalation_queue(self) -> List[RoutedRecord]:
        return self._escalation_queue

    def get_review_queue(self) -> List[RoutedRecord]:
        return self._review_queue

    def get_auto_queue(self) -> List[RoutedRecord]:
        return self._auto_queue

    def get_manifest(self) -> Dict[str, Any]:
        return self._manifest

    # ── Duplicate suppression ─────────────────────────────────────────────────

    def _suppress_consecutive_duplicates(
        self, records: List[Dict[str, Any]], window: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Suppress consecutive records where the same top anomaly rule fires
        repeatedly. Keeps the first occurrence + every Nth repeat.
        This prevents 40 near-identical imbalance warnings flooding the queue.
        """
        result      = []
        rule_streak = {}   # rule_name → consecutive count

        for record in records:
            anomalies = record.get("anomalies", [])
            if not anomalies:
                rule_streak = {}
                result.append(record)
                continue

            top_rule = anomalies[0].get("rule", "")
            count    = rule_streak.get(top_rule, 0)

            if count == 0 or count % window == 0:
                result.append(record)
                # Mark as representative if it's a repeat
                if count > 0:
                    record = dict(record)
                    record["anomaly_summary"] = (
                        f"[Repeated x{count}] " + record.get("anomaly_summary", "")
                    )
            rule_streak[top_rule] = count + 1

            # Reset streak counter for other rules
            for k in list(rule_streak.keys()):
                if k != top_rule:
                    rule_streak[k] = 0

        suppressed = len(records) - len(result)
        if suppressed:
            log.info(f"Suppressed {suppressed} consecutive duplicate records")
        return result

    # ── Routing logic ─────────────────────────────────────────────────────────

    def _route_record(
        self, record: Dict[str, Any]
    ) -> tuple:
        """
        Apply routing rules to a single record.
        Returns (RoutedRecord, was_overridden).
        """
        agent4_target = record.get("routing_target", "human_review")
        priority      = record.get("priority", "P2")
        confidence    = record.get("confidence", 0.5)
        label         = record.get("label", "normal")
        decision      = record.get("decision", "review")

        final_target    = agent4_target
        overridden      = False
        override_reason = ""

        # Rule 1: P0/P1 always → escalation regardless of anything
        if priority in self.config.always_escalate_priorities:
            if final_target != "escalation_queue":
                final_target    = "escalation_queue"
                overridden      = True
                override_reason = f"Priority {priority} always routes to escalation"

        # Rule 2: P2 always → human review (unless already escalation)
        elif priority in self.config.always_review_priorities:
            if final_target == "auto_approve":
                final_target    = "human_review"
                overridden      = True
                override_reason = f"Priority {priority} requires human review"

        # Rule 3: P3/P4 but confidence below auto-approve floor → human review
        elif (
            priority in self.config.always_auto_priorities
            and confidence < self.config.auto_approve_min_confidence
            and label != "normal"
        ):
            final_target    = "human_review"
            overridden      = True
            override_reason = (
                f"Confidence {confidence:.2f} below auto-approve floor "
                f"{self.config.auto_approve_min_confidence}"
            )

        # Rule 4: critical label always needs human — no auto-approve
        if label == "critical" and final_target == "auto_approve":
            final_target    = "human_review"
            overridden      = True
            override_reason = "Critical label cannot be auto-approved"

        routed = RoutedRecord(
            record_id           = record.get("record_id", "unknown"),
            timestamp           = record.get("timestamp", 0.0),
            routing_target      = final_target,
            priority            = priority,
            priority_int        = record.get("priority_int", 2),
            decision            = decision,
            label               = label,
            confidence          = confidence,
            confidence_band     = record.get("confidence_band", "medium"),
            max_severity        = record.get("max_severity", "none"),
            severity_score      = record.get("severity_score", 0.0),
            anomaly_summary     = record.get("anomaly_summary", ""),
            anomalies           = record.get("anomalies", []),
            recommended_actions = record.get("recommended_actions", []),
            alert_triggered     = record.get("alert_triggered", False),
            alert_message       = record.get("alert_message", ""),
            conflict            = record.get("conflict", False),
            source              = record.get("source", "unknown"),
            routed_at           = datetime.now().isoformat(),
            routing_overridden  = overridden,
            override_reason     = override_reason,
            # Sensor values
            v_terminal          = record.get("v_terminal"),
            soc                 = record.get("soc"),
            temp_c              = record.get("temp_c"),
            r0_eff              = record.get("r0_eff"),
            imbalance           = record.get("imbalance"),
            pack_soc_mean       = record.get("pack_soc_mean"),
            temp_max_c          = record.get("temp_max_c"),
        )

        return routed, overridden

    # ── Manifest + drift detection ────────────────────────────────────────────

    def _build_manifest(
        self, records: List[Dict], overrides: int
    ) -> Dict[str, Any]:
        """Build routing manifest with batch-level stats."""
        total = len(records)
        n_auto   = len(self._auto_queue)
        n_review = len(self._review_queue)
        n_esc    = len(self._escalation_queue)

        return {
            "batch_id":          self.config.batch_id,
            "source":            self.config.source,
            "total_records":     total,
            "auto_approve":      n_auto,
            "human_review":      n_review,
            "escalation_queue":  n_esc,
            "auto_approve_rate": round(n_auto / total, 3) if total else 0,
            "review_rate":       round(n_review / total, 3) if total else 0,
            "escalation_rate":   round(n_esc / total, 3) if total else 0,
            "routing_overrides": overrides,
            "alerts_fired":      sum(1 for r in self._escalation_queue if r.alert_triggered),
            "shutdown_events":   sum(1 for r in self._escalation_queue if r.decision == "shutdown"),
            "drift_alerts":      [],    # filled by _check_drift
            "routed_at":         datetime.now().isoformat(),
            "config": {
                "auto_approve_min_confidence": self.config.auto_approve_min_confidence,
                "auto_approve_floor":          self.config.auto_approve_floor,
                "escalation_ceiling":          self.config.escalation_ceiling,
            },
        }

    def _check_drift(self):
        """
        Detect distribution drift in routing outcomes.
        Fires warnings if auto-approve rate is too low or
        escalation rate is too high.
        """
        m            = self._manifest
        drift_alerts = []

        auto_rate = m["auto_approve_rate"]
        esc_rate  = m["escalation_rate"]

        if auto_rate < self.config.auto_approve_floor:
            msg = (
                f"AUTO-APPROVE RATE DRIFT: {auto_rate:.1%} below floor "
                f"{self.config.auto_approve_floor:.1%} — "
                f"pipeline may be over-flagging or data quality degraded"
            )
            drift_alerts.append(msg)
            log.warning(msg)

        if esc_rate > self.config.escalation_ceiling:
            msg = (
                f"ESCALATION RATE DRIFT: {esc_rate:.1%} above ceiling "
                f"{self.config.escalation_ceiling:.1%} — "
                f"system under stress or fault injection rate too high"
            )
            drift_alerts.append(msg)
            log.warning(msg)

        if m["shutdown_events"] > 0:
            msg = f"SHUTDOWN EVENTS: {m['shutdown_events']} shutdown decision(s) in this batch"
            drift_alerts.append(msg)
            log.warning(msg)

        m["drift_alerts"] = drift_alerts


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))

    print("=" * 65)
    print("HITL-Ops | Confidence Router")
    print("=" * 65)

    # ── Run full agent pipeline ───────────────────────────────────────────────
    try:
        from extraction_agent     import ExtractionAgent
        from anomaly_agent        import AnomalyAgent
        from classification_agent import ClassificationAgent
        from decision_agent       import DecisionAgent

        print("\n[Pipeline] Running all 4 agents...")

        records = ExtractionAgent(
            mode="pack", profile="wltp", duration_s=3600,
            sample_every=60, initial_soc=0.95,
            inject_faults=True, fault_ratio=0.25,
        ).run()

        records = AnomalyAgent(suppress_severities=["low"]).run(records)
        records = ClassificationAgent().run(records)
        records = DecisionAgent().run(records)

        print(f"  Pipeline complete — {len(records)} records ready for routing\n")

    except ImportError as e:
        print(f"\n[Pipeline] Agents not found ({e}) — using synthetic records")
        records = [
            # Auto-approve candidates
            *[{
                "record_id": f"syn-auto-{i:03d}", "timestamp": float(i * 60),
                "routing_target": "auto_approve", "priority": "P4", "priority_int": 4,
                "decision": "ignore", "label": "normal", "confidence": 0.85,
                "confidence_band": "high", "max_severity": "none", "severity_score": 0.0,
                "anomaly_summary": "No anomalies detected",
                "anomalies": [], "recommended_actions": [],
                "alert_triggered": False, "alert_message": "",
                "conflict": False, "source": "synthetic",
            } for i in range(10)],
            # Human review
            {
                "record_id": "syn-review-001", "timestamp": 600.0,
                "routing_target": "human_review", "priority": "P2", "priority_int": 2,
                "decision": "review", "label": "warning", "confidence": 0.70,
                "confidence_band": "medium", "max_severity": "medium", "severity_score": 0.5,
                "anomaly_summary": "Overvoltage warning",
                "anomalies": [{"rule": "overvoltage_warning", "severity": "medium",
                               "feature": "v_terminal", "value": 4.18,
                               "threshold": 4.15, "message": "Overvoltage warning"}],
                "recommended_actions": ["Verify sensor", "Check charge rate"],
                "alert_triggered": False, "alert_message": "",
                "conflict": True, "source": "synthetic",
            },
            # Escalation
            {
                "record_id": "syn-esc-001", "timestamp": 1200.0,
                "routing_target": "escalation_queue", "priority": "P0", "priority_int": 0,
                "decision": "shutdown", "label": "critical", "confidence": 0.98,
                "confidence_band": "high", "max_severity": "critical", "severity_score": 1.0,
                "anomaly_summary": "Thermal runaway risk detected",
                "anomalies": [{"rule": "thermal_runaway_risk", "severity": "critical",
                               "feature": "temp_c", "value": 63.0,
                               "threshold": 60.0, "message": "Thermal runaway"}],
                "recommended_actions": ["Disconnect battery", "Activate cooling"],
                "alert_triggered": True,
                "alert_message": "[P0] ALERT — SHUTDOWN | Thermal runaway risk",
                "conflict": False, "source": "synthetic",
            },
        ]

    # ── Run router ────────────────────────────────────────────────────────────
    config = RouterConfig(
        auto_approve_min_confidence=0.80,
        auto_approve_floor=0.50,
        escalation_ceiling=0.40,
    )

    router = ConfidenceRouter(config=config)
    queues = router.route(records)
    router.print_queues()

    # Save
    results_dir = str(Path(__file__).resolve().parent.parent / "results")
    paths = router.save_queues(results_dir)
    print("\n[Saved Files]")
    for name, path in paths.items():
        print(f"  {name:20s} → {path}")

    print("\n" + "=" * 65)
    print("Confidence Router — DONE.")
    print("Next: review_interface.py (human review UI)")
    print("=" * 65)
