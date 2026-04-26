# 🛠️ AI Agents Instructions (AGENTS.md)

This file describes the structure and specifics of the **Weighted Attention Message Pruner (WAMP)** project. Use it for context when working with the codebase.

---

## 📌 Project at a Glance

WAMP is a research-oriented context compression engine. It implements a **Tri-modal Adaptive Engine** that uses a hybrid intent classifier (Statistical + LogReg) to select the optimal pruning strategy for each user query.

---

## 🗂 Project Structure

```text
wamp-proxy/
├── src/
│   └── wamp/
│       ├── api/            # FastAPI endpoints and proxy logic
│       ├── core/           # Logic: WAMPruner engine, Router, Config
│       └── __init__.py
├── benchmarks/             # Research suite (Needle, Multi-Doc, Coherence)
├── tools/                  # Model Exporter, Router Trainer, HF Uploader
├── tests/                  # Unit and API tests
├── main.py                 # 🚀 Entry point
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Dev, ML, and Export tools
├── Makefile                # Developer shortcuts
└── README.md               # User and Research documentation
```

---

## 🧠 The Tri-modal Adaptive Engine

### 1. Hybrid Intent Routing
Located in `WAMPruner.classify_task`.
- **Stage 1 (Needle):** Statistical attention peak detection (Mean > 0.12).
- **Stage 2 (Reasoning):** Binary LogReg classifier on Mean-Pooled embeddings.
- **Stage 3 (Summary):** Keyword-based boost + Default fallback.

### 2. Specialized Pruning Algorithms
- **Needle:** `mean_max` (multiplier 0.98).
- **Reasoning:** `max_max` (multiplier 1.00).
- **Summary:** `mean_mean` (multiplier 0.96).

---

## 🧪 Testing & Research

### Core Research Scripts
- `benchmarks/mass_calibrate_long.py`: Ultra-granular threshold research.
- `benchmarks/router_audit.py`: Verifying the accuracy of the Hybrid Classifier.
- `benchmarks/adaptive_research.py`: Testing the final Tri-modal policy locally.

### CI/CD
- `.github/workflows/lint.yml`: Automatic Ruff linting.
- `.github/workflows/docker-build.yml`: Automated build and push to GHCR.

---

*Last Updated: April 25, 2026 (Refactored Research Version 2.0)*
