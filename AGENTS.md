# 🛠️ AI Agents Instructions (AGENTS.md)

This file describes the structure and specifics of the **Weighted Attention Message Pruner (WAMP)** project. Use it for context when working with the codebase.

---

## 📌 Project at a Glance

WAMP is a research-oriented context compression engine. It implements a **Tri-modal Adaptive Engine** that uses a high-precision composite classifier (89%+ accuracy) to select the optimal pruning strategy for each user query.

**Core Model:** `naranor/SetFit-Multilingual-ONNX-Router-V1` (INT8 ONNX) – modified to expose hidden attention states.

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

### 1. Composite Intent Routing
Located in `WAMPruner.classify_task`. Classification is performed on the **raw user query** to avoid system prompt interference.
- **Features:** 1539-dimension vector [CLS + Mean + Kurtosis + Max + Length].
- **Hierarchy:**
    - **Stage 1 (Reasoning):** Binary LogReg (vs All). High priority for logic preservation.
    - **Stage 2 (Needle):** Binary LogReg (vs All). Specific fact retrieval.
    - **Stage 3 (Summary):** Default fallback for general recap tasks.

### 2. Specialized Pruning Policies (Safe Mode)
| Category | Algorithm | Multiplier | Focus |
| :--- | :--- | :--- | :--- |
| **Reasoning** | `cls_max` | **0.92** | Logical chains & synthesis |
| **Needle** | `mean_max` | **0.98** | Pinpoint parameter retrieval |
| **Summary** | `mean_mean` | **0.90** | Global semantic compression |

---

## 📊 Performance Benchmarks (Raw Tokens)

| Mode | Scenario | Token Savings | Recall | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **SAFE** | Needle / Logic | **1-17%** | **100%** | ✅ Recommended for Agents |
| **AGGRESSIVE** | Needle / Logic | **45-50%** | **0-14%** | ❌ High risk of data loss |
| **BALANCED** | Summary | **43%** | **75%** | ⚠️ Good for general chat |

---

## 🧪 Key Research Scripts
- `tools/train_router.py`: Unified trainer for both Needle and Reasoning classifiers.
- `tools/audit_truth.py`: Honest token-based savings audit (v4).
- `benchmarks/final_validation_no_proxy.py`: Long-context (114+ msg) validation script.

---

*Last Updated: April 25, 2026 (Refactored Research Version 3.0 - Truthful Audit)*
