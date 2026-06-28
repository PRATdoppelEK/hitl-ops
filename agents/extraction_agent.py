"""
Agent 1 — Extraction Agent
HITL-Ops: Human-in-the-Loop ML Operations Platform

Role:
    Runs the battery ECM simulation (or loads real data) and extracts
    a clean, structured feature dict for each timestep. This is the
    entry point of the entire pipeline.

Output per record:
    {
        "record_id":    str,       # unique ID for this reading
        "timestamp":    float,     # simulation time [s]
        "source":       str,       # "ecm_simulation" | "pack_simulation" | "file"
        # Cell-level features
        "current_a":   float,
        "soc":         float,
        "v_terminal":  float,
        "v_ocv":       float,
        "v_rc1":       float,
        "v_rc2":       float,
        "temp_c":      float,
        "r0_eff":      float,
        # Pack-level features (None if single-cell mode)
        "pack_voltage_v":  float | None,
        "pack_soc_mean":   float | None,
        "pack_soc_min":    float | None,
        "pack_soc_max":    float | None,
        "imbalance":       float | None,
        "temp_max_c":      float | None,
        "temp_mean_c":     float | None,
        # Derived features (computed here, used by Agent 2)
        "v_drop":          float,   # OCV - V_terminal
        "dv_dt":           float,   # voltage rate of change
        "dt_dt":           float,   # temperature rate of change
        "soc_change_rate": float,   # SOC change per second
        # Metadata
        "extraction_status": str,   # "ok" | "warn" | "error"
        "notes":             list,  # any extraction warnings
    }

Author: Prateek Gaur
Project: hitl-ops
"""

import sys
import uuid
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

# ── Path setup — import from battery-ecm-simulation ──────────────────────────

ECM_SRC = Path.home() / "Desktop" / "github" / "battery-ecm-simulation" / "src"
sys.path.insert(0, str(ECM_SRC))

from ecm_model import (
    BatteryECM,
    CellParameters,
    cc_discharge_profile,
    cccv_charge_profile,
    wltp_like_profile,
)
from pack_simulation import BatteryPackSimulator, PackConfiguration

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ExtractionAgent] %(levelname)s — %(message)s",
)
log = logging.getLogger("ExtractionAgent")


# ── Agent class ───────────────────────────────────────────────────────────────

class ExtractionAgent:
    """
    Agent 1 — Data Extraction

    Responsibilities:
    - Run ECM or Pack simulations with configurable profiles
    - Accept pre-generated numpy arrays (real data)
    - Compute derived features (rates of change, voltage drop)
    - Return a clean list of feature dicts ready for Agent 2

    Usage:
        agent = ExtractionAgent(mode="pack", profile="wltp")
        records = agent.run()
    """

    PROFILES = ["cc_discharge", "cccv_charge", "wltp"]
    MODES    = ["cell", "pack"]

    def __init__(
        self,
        mode:           str   = "pack",        # "cell" or "pack"
        profile:        str   = "cc_discharge", # current profile type
        c_rate:         float = 0.5,            # C-rate for CC profiles
        duration_s:     int   = 1800,           # duration for WLTP
        dt:             float = 1.0,            # time step [s]
        sample_every:   int   = 10,             # downsample: take every Nth step
        initial_soc:    float = 1.0,
        # Pack config
        n_series:       int   = 14,
        n_parallel:     int   = 3,
        soh_spread:     float = 0.04,
        # Fault injection
        inject_faults:  bool  = False,          # inject realistic fault scenarios
        fault_ratio:    float = 0.25,           # fraction of records to corrupt
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if profile not in self.PROFILES:
            raise ValueError(f"profile must be one of {self.PROFILES}")

        self.mode          = mode
        self.profile       = profile
        self.c_rate        = c_rate
        self.duration_s    = duration_s
        self.dt            = dt
        self.sample_every  = sample_every
        self.initial_soc   = initial_soc
        self.inject_faults = inject_faults
        self.fault_ratio   = fault_ratio

        # Cell params
        self.cell_params = CellParameters(
            capacity_ah=50.0,
            r0=0.002,
            r1=0.001,
            c1=3000,
            r2=0.0008,
            c2=10000,
        )

        # Pack config
        self.pack_config = PackConfiguration(
            n_series=n_series,
            n_parallel=n_parallel,
            capacity_ah=50.0,
            soh_spread=soh_spread,
        )

        self._records: List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> List[Dict[str, Any]]:
        """
        Run the extraction pipeline and return list of feature records.
        This is what the pipeline calls.
        """
        log.info(f"Starting extraction | mode={self.mode} | profile={self.profile}")

        current_profile = self._build_current_profile()
        log.info(f"Current profile length: {len(current_profile)} steps")

        if self.mode == "cell":
            raw_records = self._run_cell_simulation(current_profile)
        else:
            raw_records = self._run_pack_simulation(current_profile)

        # Downsample
        raw_records = raw_records[::self.sample_every]
        log.info(f"After downsampling ({self.sample_every}x): {len(raw_records)} records")

        # Compute derived features
        enriched = self._compute_derived_features(raw_records)

        # Inject faults if enabled
        if self.inject_faults:
            enriched = self._inject_fault_scenarios(enriched)
            log.info(f"Fault injection enabled (ratio={self.fault_ratio})")

        # Validate and tag
        self._records = [self._validate_record(r) for r in enriched]

        ok_count   = sum(1 for r in self._records if r["extraction_status"] == "ok")
        warn_count = sum(1 for r in self._records if r["extraction_status"] == "warn")
        log.info(f"Extraction complete — {ok_count} ok, {warn_count} warnings")

        return self._records

    def run_from_arrays(
        self,
        time_s:     np.ndarray,
        current_a:  np.ndarray,
        soc:        np.ndarray,
        v_terminal: np.ndarray,
        v_ocv:      np.ndarray,
        temp_c:     np.ndarray,
        r0_eff:     np.ndarray,
        v_rc1:      Optional[np.ndarray] = None,
        v_rc2:      Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Accept pre-computed arrays (e.g. from real hardware or existing
        simulation) and run the extraction pipeline on them.
        """
        n = len(time_s)
        log.info(f"Extracting from arrays — {n} timesteps")

        raw_records = []
        for i in range(n):
            raw_records.append({
                "timestamp":   float(time_s[i]),
                "source":      "file",
                "current_a":   float(current_a[i]),
                "soc":         float(soc[i]),
                "v_terminal":  float(v_terminal[i]),
                "v_ocv":       float(v_ocv[i]),
                "temp_c":      float(temp_c[i]),
                "r0_eff":      float(r0_eff[i]),
                "v_rc1":       float(v_rc1[i]) if v_rc1 is not None else 0.0,
                "v_rc2":       float(v_rc2[i]) if v_rc2 is not None else 0.0,
                # Pack fields not available in cell-only arrays
                "pack_voltage_v": None,
                "pack_soc_mean":  None,
                "pack_soc_min":   None,
                "pack_soc_max":   None,
                "imbalance":      None,
                "temp_max_c":     None,
                "temp_mean_c":    None,
            })

        enriched = self._compute_derived_features(raw_records)
        self._records = [self._validate_record(r) for r in enriched]
        return self._records

    def save(self, path: str = "results/extraction_output.jsonl") -> str:
        """Save extracted records to JSONL file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for record in self._records:
                f.write(json.dumps(record) + "\n")
        log.info(f"Saved {len(self._records)} records → {out_path}")
        return str(out_path)

    def summary(self) -> Dict[str, Any]:
        """Return a quick summary of the extracted batch."""
        if not self._records:
            return {"status": "no_records"}

        soc_vals  = [r["soc"] for r in self._records]
        temp_vals = [r["temp_c"] for r in self._records]
        volt_vals = [r["v_terminal"] for r in self._records]

        return {
            "total_records":    len(self._records),
            "mode":             self.mode,
            "profile":          self.profile,
            "soc_range":        [round(min(soc_vals), 3), round(max(soc_vals), 3)],
            "temp_range_c":     [round(min(temp_vals), 2), round(max(temp_vals), 2)],
            "voltage_range_v":  [round(min(volt_vals), 3), round(max(volt_vals), 3)],
            "ok_records":       sum(1 for r in self._records if r["extraction_status"] == "ok"),
            "warn_records":     sum(1 for r in self._records if r["extraction_status"] == "warn"),
            "extraction_time":  datetime.now().isoformat(),
        }

    # ── Simulation runners ────────────────────────────────────────────────────

    def _run_cell_simulation(self, current_profile: np.ndarray) -> List[Dict]:
        model = BatteryECM(params=self.cell_params, extended=True, dt=self.dt)
        model.reset(initial_soc=self.initial_soc)
        raw = model.simulate(current_profile)

        records = []
        for step in raw:
            records.append({
                "timestamp":      step["time_s"],
                "source":         "ecm_simulation",
                "current_a":      step["current_a"],
                "soc":            step["soc"],
                "v_terminal":     step["v_terminal"],
                "v_ocv":          step["v_ocv"],
                "v_rc1":          step["v_rc1"],
                "v_rc2":          step["v_rc2"],
                "temp_c":         step["temp_c"],
                "r0_eff":         step["r0_eff"],
                # No pack data in cell mode
                "pack_voltage_v": None,
                "pack_soc_mean":  None,
                "pack_soc_min":   None,
                "pack_soc_max":   None,
                "imbalance":      None,
                "temp_max_c":     None,
                "temp_mean_c":    None,
            })
        return records

    def _run_pack_simulation(self, current_profile: np.ndarray) -> List[Dict]:
        pack = BatteryPackSimulator(config=self.pack_config, dt=self.dt)
        pack.reset(initial_soc=self.initial_soc, soc_spread=0.05)

        # Run pack simulation
        pack_history = pack.simulate_pack(current_profile, balance=True)

        # Also get cell-level data from first cell as representative
        cell_arrays = pack.cells[0].get_history_arrays()

        records = []
        for i, pack_step in enumerate(pack_history):
            records.append({
                "timestamp":      float(cell_arrays["time_s"][i]) if i < len(cell_arrays.get("time_s", [])) else float(i),
                "source":         "pack_simulation",
                "current_a":      float(cell_arrays["current_a"][i]) if "current_a" in cell_arrays else float(current_profile[i]) if i < len(current_profile) else 0.0,
                "soc":            float(cell_arrays["soc"][i]) if "soc" in cell_arrays else pack_step["pack_soc_mean"],
                "v_terminal":     float(cell_arrays["v_terminal"][i]) if "v_terminal" in cell_arrays else 0.0,
                "v_ocv":          float(cell_arrays["v_ocv"][i]) if "v_ocv" in cell_arrays else 0.0,
                "v_rc1":          float(cell_arrays["v_rc1"][i]) if "v_rc1" in cell_arrays else 0.0,
                "v_rc2":          float(cell_arrays["v_rc2"][i]) if "v_rc2" in cell_arrays else 0.0,
                "temp_c":         float(cell_arrays["temp_c"][i]) if "temp_c" in cell_arrays else pack_step["temp_mean_c"],
                "r0_eff":         float(cell_arrays["r0_eff"][i]) if "r0_eff" in cell_arrays else 0.002,
                # Pack-level
                "pack_voltage_v": pack_step["pack_voltage_v"],
                "pack_soc_mean":  pack_step["pack_soc_mean"],
                "pack_soc_min":   pack_step["pack_soc_min"],
                "pack_soc_max":   pack_step["pack_soc_max"],
                "imbalance":      pack_step["imbalance"],
                "temp_max_c":     pack_step["temp_max_c"],
                "temp_mean_c":    pack_step["temp_mean_c"],
            })
        return records

    # ── Fault injection ───────────────────────────────────────────────────────

    def _inject_fault_scenarios(self, records: List[Dict]) -> List[Dict]:
        """
        Inject realistic battery fault scenarios into a subset of records.
        Simulates: thermal event, deep discharge, aging, cell imbalance,
        voltage spike, rapid temp rise — mixed into otherwise clean data.
        """
        rng = np.random.default_rng(seed=42)
        n   = len(records)

        # Pick fault indices — spread across the batch, not all at the start
        n_faults     = max(1, int(n * self.fault_ratio))
        fault_indices = sorted(rng.choice(n, size=n_faults, replace=False))

        # Fault scenario pool — each is a dict of field overrides
        fault_scenarios = [
            # Thermal warning
            {
                "temp_c": 48.5, "temp_max_c": 51.0, "temp_mean_c": 48.5,
                "dt_dt": 0.062, "_fault_tag": "thermal_warning",
            },
            # Thermal runaway risk
            {
                "temp_c": 63.0, "temp_max_c": 67.0, "temp_mean_c": 63.0,
                "dt_dt": 0.12, "_fault_tag": "thermal_runaway_risk",
            },
            # Deep discharge
            {
                "soc": 0.03, "pack_soc_mean": 0.03, "pack_soc_min": 0.01,
                "v_terminal": 2.91, "v_ocv": 3.05, "v_drop": 0.14,
                "_fault_tag": "deep_discharge",
            },
            # Low SOC warning
            {
                "soc": 0.08, "pack_soc_mean": 0.08, "pack_soc_min": 0.06,
                "v_terminal": 3.18, "v_ocv": 3.22,
                "_fault_tag": "low_soc",
            },
            # Cell imbalance — balancing failure
            {
                "imbalance": 0.09, "pack_soc_min": 0.12, "pack_soc_max": 0.68,
                "_fault_tag": "cell_imbalance_critical",
            },
            # Aging — resistance growth
            {
                "r0_eff": 0.0052, "v_drop": 0.18,
                "_fault_tag": "resistance_growth",
            },
            # Rapid voltage drop — load spike / connection fault
            {
                "v_terminal": 3.31, "dv_dt": -0.009,
                "_fault_tag": "rapid_voltage_drop",
            },
            # Overvoltage — charging fault
            {
                "v_terminal": 4.27, "v_ocv": 4.28, "v_drop": 0.01,
                "current_a": -50.0, "_fault_tag": "overvoltage",
            },
            # Combined: aging + thermal
            {
                "r0_eff": 0.0048, "temp_c": 47.0, "temp_max_c": 49.5,
                "v_drop": 0.17, "_fault_tag": "aging_thermal_combined",
            },
        ]

        result = []
        fault_pool_size = len(fault_scenarios)
        fault_set       = set(fault_indices)

        for i, record in enumerate(records):
            if i not in fault_set:
                result.append(record)
                continue

            # Pick a fault scenario (cycle through pool)
            scenario = fault_scenarios[i % fault_pool_size]
            r        = dict(record)

            # Apply overrides
            for key, val in scenario.items():
                if key.startswith("_"):
                    continue
                if key in r:
                    r[key] = val

            r["source"]     = f"pack_simulation+fault:{scenario['_fault_tag']}"
            r["_fault_tag"] = scenario["_fault_tag"]

            # Recompute v_drop if v_ocv and v_terminal were both overridden
            if "v_ocv" in scenario and "v_terminal" in scenario:
                r["v_drop"] = round(r["v_ocv"] - r["v_terminal"], 6)

            result.append(r)

        injected = sum(1 for r in result if "_fault_tag" in r)
        log.info(f"Injected {injected} fault records across {n} total")
        return result

    # ── Feature engineering ───────────────────────────────────────────────────

    def _compute_derived_features(self, records: List[Dict]) -> List[Dict]:
        """
        Compute derived features that Agent 2 (anomaly detection) needs.
        These require looking at consecutive records.
        """
        enriched = []
        for i, r in enumerate(records):
            r = dict(r)  # copy

            # Unique ID for this record
            r["record_id"] = str(uuid.uuid4())[:12]

            # Voltage drop = OCV - V_terminal (internal loss)
            r["v_drop"] = round(r["v_ocv"] - r["v_terminal"], 6)

            # Rate of change features (zero for first record)
            if i == 0:
                r["dv_dt"]           = 0.0
                r["dt_dt"]           = 0.0
                r["soc_change_rate"] = 0.0
            else:
                prev = records[i - 1]
                dt   = r["timestamp"] - prev["timestamp"]
                if dt > 0:
                    r["dv_dt"]           = round((r["v_terminal"] - prev["v_terminal"]) / dt, 6)
                    r["dt_dt"]           = round((r["temp_c"]     - prev["temp_c"])     / dt, 6)
                    r["soc_change_rate"] = round((r["soc"]        - prev["soc"])        / dt, 6)
                else:
                    r["dv_dt"] = r["dt_dt"] = r["soc_change_rate"] = 0.0

            enriched.append(r)
        return enriched

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_record(self, r: Dict) -> Dict:
        """Tag records with extraction_status and any notes."""
        r = dict(r)
        notes  = []
        status = "ok"

        # Check for NaN / Inf
        for key in ["v_terminal", "v_ocv", "soc", "temp_c", "r0_eff"]:
            val = r.get(key)
            if val is None:
                continue
            if np.isnan(val) or np.isinf(val):
                notes.append(f"invalid value in {key}: {val}")
                status = "warn"

        # Check physically reasonable ranges
        if not (0.0 <= r["soc"] <= 1.0):
            notes.append(f"SOC out of range: {r['soc']:.3f}")
            status = "warn"

        if not (2.5 <= r["v_terminal"] <= 4.5):
            notes.append(f"V_terminal unusual: {r['v_terminal']:.3f} V")
            status = "warn"

        if r["temp_c"] > 80 or r["temp_c"] < -20:
            notes.append(f"Temperature unusual: {r['temp_c']:.1f} °C")
            status = "warn"

        r["extraction_status"] = status
        r["notes"]             = notes
        return r

    # ── Current profile builder ───────────────────────────────────────────────

    def _build_current_profile(self) -> np.ndarray:
        capacity = self.cell_params.capacity_ah
        if self.profile == "cc_discharge":
            return cc_discharge_profile(capacity, self.c_rate, self.dt)
        elif self.profile == "cccv_charge":
            return cccv_charge_profile(capacity, self.dt)
        elif self.profile == "wltp":
            return wltp_like_profile(capacity, self.duration_s, self.dt)
        else:
            raise ValueError(f"Unknown profile: {self.profile}")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("HITL-Ops | Agent 1 — Extraction Agent")
    print("=" * 60)

    # Mixed mode: normal pack + WLTP stress + fault injection
    print("\n[Test] Pack simulation — WLTP + fault injection (25%)")
    agent = ExtractionAgent(
        mode="pack",
        profile="wltp",
        duration_s=3600,
        sample_every=60,
        initial_soc=0.95,
        inject_faults=True,
        fault_ratio=0.25,
    )
    records = agent.run()
    summary = agent.summary()

    print(f"\n  Summary:")
    for k, v in summary.items():
        print(f"    {k:22s}: {v}")

    # Show first record structure
    print(f"\n  Sample record (first):")
    first = records[0]
    for k, v in first.items():
        print(f"    {k:22s}: {v}")

    # Show fault records only
    faults = [r for r in records if "_fault_tag" in r]
    print(f"\n  Fault records injected: {len(faults)}")
    for r in faults[:3]:
        print(f"    [{r['_fault_tag']:30s}] soc={r['soc']:.3f} | temp={r['temp_c']:.1f}°C | v={r['v_terminal']:.3f}V | r0={r['r0_eff']:.5f}Ω")

    # Save output
    path = agent.save("results/extraction_output.jsonl")
    print(f"\n  Saved → {path}")

    print("\n" + "=" * 60)
    print("Agent 1 — DONE. Passing to Agent 2.")
    print("=" * 60)
