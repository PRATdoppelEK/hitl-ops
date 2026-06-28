"""
Feedback Loop — RLHF Dataset Builder + Exporter
HITL-Ops: Human-in-the-Loop ML Operations Platform

Role:
    Reads human review decisions from review_interface.py output
    (results/human_reviews.jsonl) and transforms them into a
    structured RLHF-format training dataset.

    This closes the loop:
        Human corrections → structured training data → model improves

    Two components:
        DatasetBuilder  → processes raw human reviews into training pairs
        RLHFExporter    → exports in multiple formats (JSONL, CSV, HF-ready)

    RLHF training record format:
        {
            "prompt":       str,   # the input context shown to the model
            "chosen":       str,   # the correct/preferred output (human label)
            "rejected":     str,   # the wrong output (model's original label)
            "reward":       float, # quality_rating normalized to 0-1
            "was_corrected":bool,  # True if human changed the label
            "metadata":     dict,  # record_id, timestamp, anomaly details
        }

Author: Prateek Gaur
Project: hitl-ops
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FeedbackLoop] %(levelname)s — %(message)s",
)
log = logging.getLogger("FeedbackLoop")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent
RESULTS_DIR   = BASE_DIR / "results"
REVIEWS_FILE  = RESULTS_DIR / "human_reviews.jsonl"
RLHF_OUT      = RESULTS_DIR / "rlhf_dataset.jsonl"
CSV_OUT       = RESULTS_DIR / "rlhf_dataset.csv"
STATS_OUT     = RESULTS_DIR / "feedback_stats.json"


# ── Dataset Builder ───────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Reads human_reviews.jsonl and builds structured RLHF training pairs.

    Each human review becomes one training record:
    - If the human corrected the label → strong training signal
    - If the human approved           → weaker positive signal
    - Quality rating (1-5)            → reward signal
    """

    def __init__(self, reviews_file: Path = REVIEWS_FILE):
        self.reviews_file = reviews_file
        self._records: List[Dict[str, Any]] = []
        self._dataset: List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> List[Dict[str, Any]]:
        """
        Load reviews and build the RLHF dataset.
        Returns list of training records.
        """
        self._records = self._load_reviews()

        if not self._records:
            log.warning(f"No reviews found at {self.reviews_file}")
            log.warning("Run review_interface.py and submit some reviews first.")
            return []

        log.info(f"Loaded {len(self._records)} human reviews")
        self._dataset = [self._build_record(r) for r in self._records]

        # Stats
        corrections = sum(1 for r in self._dataset if r["was_corrected"])
        approvals   = sum(1 for r in self._dataset if not r["was_corrected"])
        avg_reward  = float(np.mean([r["reward"] for r in self._dataset]))

        log.info(
            f"Dataset built — {len(self._dataset)} records | "
            f"corrections={corrections} | approvals={approvals} | "
            f"avg_reward={avg_reward:.3f}"
        )
        return self._dataset

    def summary(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        if not self._dataset:
            return {"status": "empty"}

        rewards      = [r["reward"] for r in self._dataset]
        corrections  = [r for r in self._dataset if r["was_corrected"]]
        label_pairs  = {}

        for r in corrections:
            pair = f"{r['metadata']['original_label']} → {r['chosen']}"
            label_pairs[pair] = label_pairs.get(pair, 0) + 1

        return {
            "total_records":      len(self._dataset),
            "corrections":        len(corrections),
            "approvals":          len(self._dataset) - len(corrections),
            "correction_rate":    round(len(corrections) / len(self._dataset), 3),
            "avg_reward":         round(float(np.mean(rewards)), 3),
            "min_reward":         round(float(np.min(rewards)), 3),
            "max_reward":         round(float(np.max(rewards)), 3),
            "label_corrections":  label_pairs,
            "built_at":           datetime.now().isoformat(),
        }

    # ── Core builder ─────────────────────────────────────────────────────────

    def _build_record(self, review: Dict) -> Dict[str, Any]:
        """Convert one human review into an RLHF training record."""

        original_label  = review.get("original_label", "unknown")
        corrected_label = review.get("corrected_label", original_label)
        quality_rating  = int(review.get("quality_rating", 3))
        was_corrected   = review.get("was_corrected", False)
        anomaly_summary = review.get("anomaly_summary", "")
        anomalies       = review.get("anomalies", [])
        action          = review.get("human_action", "approve")
        confidence      = review.get("confidence", 0.5)
        severity_score  = review.get("severity_score", 0.0)
        priority        = review.get("priority", "P2")

        # Build prompt — the context the model saw
        anomaly_text = self._format_anomalies(anomalies)
        prompt = (
            f"Battery ECM sensor reading analysis:\n"
            f"Summary: {anomaly_summary}\n"
            f"Detected anomalies:\n{anomaly_text}\n"
            f"Model confidence: {confidence:.2f}\n"
            f"Severity score: {severity_score:.2f}\n"
            f"Priority: {priority}\n"
            f"Classify this reading as: normal, warning, or critical."
        )

        # Chosen = what the human said is correct
        chosen   = self._label_to_response(corrected_label, action)

        # Rejected = what the model originally said (if different)
        rejected = self._label_to_response(original_label, "model_prediction")

        # Reward = normalized quality rating
        # 5 stars → 1.0, 1 star → 0.0
        # Corrections with low rating are extra penalized
        base_reward = (quality_rating - 1) / 4.0
        if was_corrected and quality_rating <= 2:
            reward = base_reward * 0.5   # model was clearly wrong
        elif not was_corrected and quality_rating >= 4:
            reward = min(base_reward * 1.2, 1.0)  # model was clearly right
        else:
            reward = base_reward

        return {
            "prompt":        prompt,
            "chosen":        chosen,
            "rejected":      rejected,
            "reward":        round(float(reward), 4),
            "was_corrected": was_corrected,
            "metadata": {
                "record_id":      review.get("record_id", "unknown"),
                "reviewed_at":    review.get("reviewed_at", ""),
                "original_label": original_label,
                "human_action":   action,
                "quality_rating": quality_rating,
                "priority":       priority,
                "decision":       review.get("decision", ""),
                "comment":        review.get("comment", ""),
                "anomaly_count":  len(anomalies),
            },
        }

    def _format_anomalies(self, anomalies: List[Dict]) -> str:
        if not anomalies:
            return "  - No anomalies detected"
        lines = []
        for a in anomalies:
            lines.append(
                f"  - [{a.get('severity','?').upper()}] {a.get('rule','?')}: "
                f"{a.get('message','')}"
            )
        return "\n".join(lines)

    def _label_to_response(self, label: str, source: str) -> str:
        descriptions = {
            "normal":   "Classification: NORMAL — No action required. "
                        "All parameters within acceptable range.",
            "warning":  "Classification: WARNING — Monitor closely. "
                        "Parameters approaching threshold limits.",
            "critical": "Classification: CRITICAL — Immediate action required. "
                        "Parameters outside safe operating range.",
            "unknown":  "Classification: UNKNOWN — Insufficient data to classify.",
        }
        base = descriptions.get(label, descriptions["unknown"])
        return f"{base} [Source: {source}]"

    def _load_reviews(self) -> List[Dict]:
        if not self.reviews_file.exists():
            return []
        records = []
        with open(self.reviews_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records


# ── RLHF Exporter ─────────────────────────────────────────────────────────────

class RLHFExporter:
    """
    Exports the RLHF dataset in multiple formats:
        - JSONL  → standard format for most training pipelines
        - CSV    → for spreadsheet analysis
        - HF     → Hugging Face datasets-compatible format
        - Stats  → summary statistics JSON
    """

    def __init__(self, dataset: List[Dict[str, Any]]):
        self.dataset = dataset

    def export_jsonl(self, path: Path = RLHF_OUT) -> str:
        """Export as JSONL — one record per line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in self.dataset:
                f.write(json.dumps(record) + "\n")
        log.info(f"JSONL exported → {path} ({len(self.dataset)} records)")
        return str(path)

    def export_csv(self, path: Path = CSV_OUT) -> str:
        """Export as CSV — flattened for analysis."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.dataset:
            return str(path)

        # Flatten metadata fields
        rows = []
        for r in self.dataset:
            row = {
                "prompt":          r["prompt"][:200],   # truncate for CSV
                "chosen":          r["chosen"],
                "rejected":        r["rejected"],
                "reward":          r["reward"],
                "was_corrected":   r["was_corrected"],
            }
            row.update(r["metadata"])
            rows.append(row)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        log.info(f"CSV exported → {path} ({len(rows)} rows)")
        return str(path)

    def export_hf_format(self, path: Optional[Path] = None) -> str:
        """
        Export in Hugging Face TRL/RLHF format.
        Compatible with trl.PPOTrainer and trl.RewardTrainer.
        """
        if path is None:
            path = RESULTS_DIR / "rlhf_dataset_hf.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for r in self.dataset:
                hf_record = {
                    "input_ids":    r["prompt"],        # tokenizer handles conversion
                    "chosen":       r["chosen"],
                    "rejected":     r["rejected"],
                    "score_chosen": r["reward"],
                    "score_rejected": max(0.0, r["reward"] - 0.3),
                    "label":        r["metadata"]["original_label"],
                }
                f.write(json.dumps(hf_record) + "\n")

        log.info(f"HF format exported → {path}")
        return str(path)

    def export_stats(self, path: Path = STATS_OUT) -> str:
        """Export dataset statistics."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.dataset:
            stats = {"status": "empty", "exported_at": datetime.now().isoformat()}
        else:
            rewards     = [r["reward"] for r in self.dataset]
            corrections = [r for r in self.dataset if r["was_corrected"]]
            ratings     = [r["metadata"]["quality_rating"] for r in self.dataset]

            # Label distribution
            orig_labels = {}
            corr_labels = {}
            for r in self.dataset:
                ol = r["metadata"]["original_label"]
                cl = r["chosen"].split("Classification: ")[1].split(" —")[0].lower() \
                     if "Classification:" in r["chosen"] else "unknown"
                orig_labels[ol] = orig_labels.get(ol, 0) + 1
                corr_labels[cl] = corr_labels.get(cl, 0) + 1

            stats = {
                "total_records":        len(self.dataset),
                "corrections":          len(corrections),
                "correction_rate":      round(len(corrections) / len(self.dataset), 3),
                "reward_mean":          round(float(np.mean(rewards)), 3),
                "reward_std":           round(float(np.std(rewards)), 3),
                "reward_min":           round(float(np.min(rewards)), 3),
                "reward_max":           round(float(np.max(rewards)), 3),
                "avg_quality_rating":   round(float(np.mean(ratings)), 2),
                "original_label_dist":  orig_labels,
                "corrected_label_dist": corr_labels,
                "exported_at":          datetime.now().isoformat(),
            }

        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        log.info(f"Stats exported → {path}")
        return str(path)

    def export_all(self) -> Dict[str, str]:
        """Export all formats at once."""
        return {
            "jsonl":  self.export_jsonl(),
            "csv":    self.export_csv(),
            "hf":     self.export_hf_format(),
            "stats":  self.export_stats(),
        }


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("HITL-Ops | Feedback Loop — RLHF Dataset Builder")
    print("=" * 60)

    # If no real reviews exist yet, inject sample ones for demo
    if not REVIEWS_FILE.exists() or REVIEWS_FILE.stat().st_size == 0:
        log.info("No reviews found — injecting sample reviews for demo")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        sample_reviews = [
            {
                "record_id": "demo-001", "human_action": "approve",
                "corrected_label": "critical", "original_label": "critical",
                "quality_rating": 5, "comment": "Correct — clear thermal runaway",
                "was_corrected": False, "priority": "P0", "decision": "shutdown",
                "anomaly_summary": "Temperature 63.0°C exceeds 60.0°C — thermal runaway risk",
                "anomalies": [
                    {"rule": "thermal_runaway_risk", "severity": "critical",
                     "feature": "temp_c", "value": 63.0, "threshold": 60.0,
                     "message": "Temperature 63.0°C exceeds 60.0°C — thermal runaway risk"},
                ],
                "severity_score": 1.0, "confidence": 0.98,
                "reviewed_at": datetime.now().isoformat(), "reviewer": "human_engineer",
            },
            {
                "record_id": "demo-002", "human_action": "correct",
                "corrected_label": "warning", "original_label": "critical",
                "quality_rating": 2, "comment": "Model over-classified — sensor noise",
                "was_corrected": True, "priority": "P1", "decision": "escalate",
                "anomaly_summary": "Rapid voltage drop -0.009V/s — possible fault",
                "anomalies": [
                    {"rule": "rapid_voltage_drop", "severity": "high",
                     "feature": "dv_dt", "value": -0.009, "threshold": -0.005,
                     "message": "Rapid voltage drop — possible fault or load spike"},
                ],
                "severity_score": 0.75, "confidence": 0.57,
                "reviewed_at": datetime.now().isoformat(), "reviewer": "human_engineer",
            },
            {
                "record_id": "demo-003", "human_action": "correct",
                "corrected_label": "critical", "original_label": "warning",
                "quality_rating": 1, "comment": "Model missed deep discharge — dangerous",
                "was_corrected": True, "priority": "P2", "decision": "review",
                "anomaly_summary": "Low SOC 0.08 approaching threshold",
                "anomalies": [
                    {"rule": "low_soc_warning", "severity": "medium",
                     "feature": "soc", "value": 0.08, "threshold": 0.10,
                     "message": "SOC approaching low threshold"},
                ],
                "severity_score": 0.5, "confidence": 0.75,
                "reviewed_at": datetime.now().isoformat(), "reviewer": "human_engineer",
            },
            {
                "record_id": "demo-004", "human_action": "approve",
                "corrected_label": "normal", "original_label": "normal",
                "quality_rating": 4, "comment": "",
                "was_corrected": False, "priority": "P4", "decision": "ignore",
                "anomaly_summary": "No anomalies detected",
                "anomalies": [],
                "severity_score": 0.0, "confidence": 0.85,
                "reviewed_at": datetime.now().isoformat(), "reviewer": "human_engineer",
            },
            {
                "record_id": "demo-005", "human_action": "approve",
                "corrected_label": "critical", "original_label": "critical",
                "quality_rating": 4, "comment": "Correct — cell imbalance confirmed",
                "was_corrected": False, "priority": "P1", "decision": "escalate",
                "anomaly_summary": "Cell SOC imbalance σ=0.09 — balancing failure",
                "anomalies": [
                    {"rule": "cell_imbalance_critical", "severity": "critical",
                     "feature": "imbalance", "value": 0.09, "threshold": 0.08,
                     "message": "Cell SOC imbalance σ=0.09 exceeds critical threshold"},
                ],
                "severity_score": 1.0, "confidence": 0.75,
                "reviewed_at": datetime.now().isoformat(), "reviewer": "human_engineer",
            },
        ]
        with open(REVIEWS_FILE, "w") as f:
            for r in sample_reviews:
                f.write(json.dumps(r) + "\n")
        print(f"  Sample reviews written → {REVIEWS_FILE}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("\n[Step 1] Building RLHF dataset from human reviews...")
    builder = DatasetBuilder()
    dataset = builder.build()

    if not dataset:
        print("  No dataset built — check reviews file.")
        exit(0)

    # Print summary
    summary = builder.summary()
    print(f"\n  Dataset Summary:")
    for k, v in summary.items():
        print(f"    {k:25s}: {v}")

    # Print sample record
    print(f"\n  Sample RLHF record (record 1 of {len(dataset)}):")
    print("-" * 60)
    sample = dataset[0]
    print(f"  PROMPT:\n{sample['prompt']}\n")
    print(f"  CHOSEN:\n{sample['chosen']}\n")
    print(f"  REJECTED:\n{sample['rejected']}\n")
    print(f"  REWARD:       {sample['reward']}")
    print(f"  CORRECTED:    {sample['was_corrected']}")

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"\n[Step 2] Exporting dataset...")
    exporter = RLHFExporter(dataset)
    paths    = exporter.export_all()

    print(f"\n  Exported files:")
    for fmt, path in paths.items():
        print(f"    {fmt:8s} → {path}")

    print("\n" + "=" * 60)
    print("Feedback Loop — DONE.")
    print(f"RLHF dataset ready: {len(dataset)} training records")
    print("Next: dashboard/app.py (Streamlit monitoring)")
    print("=" * 60)
