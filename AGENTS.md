# 🛠️ AI Agents Instructions (AGENTS.md)

This file describes the structure and specifics of the **Weighted Attention Message Pruner (WAMP)** project. Use it for context when working with the codebase.

---

## 📌 Project at a Glance

WAMP is a research-oriented context compression engine. It implements a **Tri-modal Adaptive Engine (V4)** that uses a high-precision **SetFit** semantic router (100% accuracy on universal benchmarks) to select the optimal pruning strategy for each user query.

**Core Model:** `naranor/SetFit-Multilingual-ONNX-Router-V1` (INT8 ONNX) – a unified engine for both intent routing and attention-based filtering.

---

## 🗂 Project Structure

```text
wamp-proxy/
├── src/
│   └── wamp/
│       ├── api/            # FastAPI endpoints and proxy logic
│       ├── core/           # Logic: WAMPruner engine (SetFit), Config
│       └── __init__.py
├── benchmarks/             # Final validation suite (Needle, Multi-Doc, Coherence)
├── tools/                  # SetFit Trainer, ONNX Exporter, Quantizer, HF Uploader
├── tests/                  # Unit and API tests
├── main.py                 # 🚀 Entry point
├── requirements.txt        # Runtime dependencies
├── Makefile                # Developer shortcuts
└── README.md               # User and Research documentation
```

---

## 🧠 The Tri-modal Adaptive Engine (V4)

### 1. SetFit Semantic Routing
Located in `WAMPruner.classify_task`. Classification uses **Mean Pooling** of hidden states followed by a Logistic Regression head.
- **Features:** High-dimensional semantic embeddings (MiniLM-L12).
- **Universality:** Trained on technical, medical, historical, and philosophical datasets (Bilingual RU/EN).
- **Categories:** Summary (0), Needle (1), Reasoning (2).

### 2. Specialized Pruning Policies (Safe Mode)
Configurable via `.env`. Default algorithm: `cls_max`.

| Category | Algorithm | Multiplier | Focus |
| :--- | :--- | :--- | :--- |
| **Needle** | `cls_max` | **0.99** | Pinpoint fact preservation (28% savings) |
| **Reasoning** | `cls_max` | **0.95** | Zero-loss logical chains (0% savings) |
| **Summary** | `cls_max` | **0.99** | High-level recap (37% savings) |

---

## 📊 Performance Benchmarks (SetFit INT8)

| Mode | Scenario | Token Savings | Recall | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **SAFE** | Needle (Facts) | **~29%** | **100%** | ✅ Recommended |
| **SAFE** | Reasoning (Logic) | **0%** | **100%** | ✅ Safe for Agents |
| **BALANCED** | Summary | **~37%** | **75%+** | ⚠️ Optimized |

---

## 🧪 Key Operational Tools
- `tools/train_setfit_router.py`: Re-train the semantic intent classifier.
- `tools/quantize_setfit_onnx.py`: Convert FP32 ONNX to efficient INT8.
- `benchmarks/final_validation_no_proxy.py`: Standalone validation of the SetFit engine.

---

*Last Updated: May 4, 2026 (Unified SetFit Engine v4.0 - 100% Recall Update)*
