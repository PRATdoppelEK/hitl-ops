# HITL-Ops 🔁

> **Human-in-the-Loop ML Operations Platform**
> End-to-end pipeline where humans and AI collaborate — agents analyze, humans decide, models improve.

---

## 🎯 What This Is

HITL-Ops is a production-grade ML operations system that combines **4 core ML engineering concepts** into one cohesive platform:

| Concept | Component |
|---|---|
| Multi-Agent Pipeline | 4 specialized agents with HITL checkpoints |
| Battery Anomaly Detection | Real ECM sensor data — voltage, SOC, temperature |
| ML Quality Gate | Human approval before any model deployment |
| RLHF Feedback Loop | Human corrections → training dataset → model improves |

---

## 🏗️ Architecture

```
DATA IN (Battery Sensors / ECM Simulation)
        ↓
MULTI-AGENT PIPELINE
  Agent 1 — Extraction       → structured features
  Agent 2 — Anomaly Detection → flags + severity
  Agent 3 — Classification    → normal / warning / critical
  Agent 4 — Decision          → ignore / review / escalate
        ↓
CONFIDENCE ROUTER
  High   → auto-approve
  Medium → human review
  Low    → escalate + alert
        ↓
HUMAN REVIEW INTERFACE (browser UI)
  Review · Correct · Rate (1–5 for RLHF)
        ↓
QUALITY GATE
  Model metrics evaluated
  Human approves or rejects deployment
  Auto-rollback on rejection
        ↓
FEEDBACK LOOP
  Corrections → RLHF dataset
  Model retrains and improves
        ↓
AUDIT + DASHBOARD
  Full audit trail · Streamlit monitoring
```

---

## 📦 Project Structure

```
hitl-ops/
├── agents/
│   ├── extraction_agent.py       # Agent 1 — ECM data extraction + feature engineering
│   ├── anomaly_agent.py          # Agent 2 — anomaly detection (voltage, temp, SOC)
│   ├── classification_agent.py   # Agent 3 — ML classification
│   └── decision_agent.py         # Agent 4 — decision recommendation
│
├── hitl/
│   ├── confidence_router.py      # Routes by confidence threshold
│   ├── review_interface.py       # Browser-based human review UI
│   └── feedback_collector.py     # Stores human ratings + corrections
│
├── quality_gate/
│   ├── model_evaluator.py        # Evaluates model metrics
│   ├── deployment_gate.py        # Human approval before deploy
│   └── rollback.py               # Auto-rollback on rejection
│
├── feedback_loop/
│   ├── dataset_builder.py        # Builds training data from corrections
│   └── rlhf_exporter.py          # Exports RLHF-format dataset
│
├── dashboard/
│   └── app.py                    # Streamlit monitoring dashboard
│
├── workflows/
│   └── hitl_ops_full.json        # Complete n8n workflow
│
├── audit/
│   └── audit_logger.py           # Full audit trail
│
├── data/
│   ├── battery_sensors/          # Sample battery sensor data
│   └── sample_models/            # Sample ML models for quality gate
│
├── results/
│   ├── approved/
│   ├── rejected/
│   ├── audit_trail.json
│   └── rlhf_dataset.jsonl
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Agents & Logic | Pure Python + scikit-learn |
| Deep Learning | PyTorch 2.6 (MPS — Apple M2) |
| LLM | Ollama — mistral + nomic-embed-text |
| Vector Store | FAISS |
| Human Review UI | Python (browser-based) |
| Orchestration | n8n |
| Dashboard | Streamlit |
| Data Source | battery-ecm-simulation (ECM model) |

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/PRATdoppelEK/hitl-ops.git
cd hitl-ops

# Install dependencies
pip install -r requirements.txt

# Run Agent 1 — Extraction
python agents/extraction_agent.py

# Run the full pipeline (coming soon)
python run_pipeline.py
```

---

## 📊 Domain: Battery ECM Data

The anomaly detection pipeline uses real Equivalent Circuit Model (ECM) outputs:

| Feature | Description | Anomaly Threshold |
|---|---|---|
| `v_terminal` | Terminal voltage | < 3.0V or > 4.25V |
| `temp_c` | Cell temperature | > 45°C warning, > 60°C critical |
| `soc` | State of charge | < 0.05 deep discharge |
| `imbalance` | SOC spread across cells | > 0.05 |
| `r0_eff` | Internal resistance | > 0.004 Ω aging signal |
| `dv_dt` | Voltage rate of change | rapid drop → fault |

---

## 🔄 HITL Philosophy

This project treats humans not as a bottleneck but as a **signal source**:
- Every human correction improves the model via RLHF
- Nothing deploys without explicit human sign-off
- Every decision — agent or human — is fully audited

---

## 📅 Build Status

| Component | Status |
|---|---|
| Agent 1 — Extraction | ✅ Done |
| Agent 2 — Anomaly Detection | ✅ Done |
| Agent 3 — Classification | ✅ Done |
| Agent 4 — Decision | ✅ Done |
| Confidence Router | ✅ Done |
| Human Review Interface | ✅ Done |
| Quality Gate | ✅ Done |
| Feedback Loop + RLHF | ✅ Done |
| Streamlit Dashboard | ✅ Done |
| n8n Workflow | ✅ Done |

---

## 👤 Author

**Prateek Gaur** — ML Engineer, Munich
GitHub: [@PRATdoppelEK](https://github.com/PRATdoppelEK)

---

## 📄 License

MIT
